import info
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

def get_env_variable(name):
  value = os.environ.get(name)
  if value is None:
    print(f'Error: Environment variable {name} not set.')
    sys.exit(1)
  return value

OUTPUT_PATH = get_env_variable('OTHELLO_OUTPUT_PATH')

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

def get_files_from_directory(directory):
  return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

models_directory = f'{OUTPUT_PATH}/models/eOthello-1/'
model_load_paths = get_files_from_directory(models_directory)
app = OracleDeployment.bind(model_load_paths=model_load_paths)

# Deploy the application locally.
ray.serve.run(app)
ray.serve.run(OracleDeployment.options(num_replicas=8).bind(model_load_paths=model_load_paths))

# This will keep the script running indefinitely
try:
  while True:
    time.sleep(1)
except KeyboardInterrupt:
  print('Shutting down...')
