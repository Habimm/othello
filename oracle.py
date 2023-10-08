import _othello_environment
import json
import numpy
import os
import ray.serve
import redis
import starlette.requests
import sys
import tensorflow
import time
import typing
import variables_info

GENERATOR_PATH = _othello_environment.parameter('OTHELLO_GENERATOR_PATH')

@ray.serve.deployment(route_prefix='/predict')
class OracleDeployment:
  def __init__(self, model_load_paths: str):
    self.oracles = {}
    for model_load_path in model_load_paths:
      self.oracles[model_load_path] = tensorflow.keras.models.load_model(model_load_path)
    self.redis_store = redis.StrictRedis(host='localhost', port=6379, db=0)
    self.redis_store.flushdb()

  async def __call__(self, request: starlette.requests.Request) -> typing.Dict:
    try:
      oracle_command = await request.json()

      assert 'board' in oracle_command
      assert 'model_load_path' in oracle_command

      board = oracle_command['board']
      model_load_path = oracle_command['model_load_path']

      # Try to retrieve from Redis cache.
      cache_key = json.dumps(board)
      cache_key = cache_key.encode('utf-8')
      cached_prediction = self.redis_store.get(cache_key)
      if cached_prediction:
        cached_prediction = numpy.frombuffer(cached_prediction, dtype=numpy.float32)
        cached_prediction = cached_prediction.reshape(1, 1)
        assert cached_prediction.shape == (1, 1), 'Assertion failed: cached_prediction.shape == (1, 1)'
        return {'exception': None, 'prediction': cached_prediction}

      if model_load_path not in self.oracles:
        self.oracles[model_load_path] = tensorflow.keras.models.load_model(model_load_path)

      assert model_load_path in self.oracles, 'Assertion failed: model_load_path in self.oracles'
      prediction = self.oracles[model_load_path].predict([board], verbose=0)

      # Commit to Redis cache.
      cached_prediction = prediction.tobytes()
      self.redis_store.set(cache_key, cached_prediction)

      assert prediction.shape == (1, 1), 'Assertion failed: prediction.shape == (1, 1)'
      return {'exception': None, 'prediction': prediction}

    except Exception as e:
      return {
        'exception': f'{e}',
      }

def oracle(model_name):
  app = OracleDeployment.bind(model_load_paths=[GENERATOR_PATH])

  # Deploy the application locally.
  ray.serve.run(app)
  ray.serve.run(OracleDeployment.options(num_replicas=1).bind(model_load_paths=[GENERATOR_PATH]))

if __name__ == '__main__':
  oracle(
    model_name = 'eOthello-1',
  )

  # This will keep the script running indefinitely
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    print('Shutting down...')
