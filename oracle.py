import info
import json
import numpy
import ray.serve
import redis
import starlette.requests
import tensorflow
import time
import typing

@ray.serve.deployment(route_prefix="/predict")
class OracleDeployment:
  def __init__(self, oracle_path: str):
    self.oracle = tensorflow.keras.models.load_model(oracle_path)
    self.redis_store = redis.StrictRedis(host='localhost', port=6379, db=0)

  async def __call__(self, request: starlette.requests.Request) -> typing.Dict:
    board = await request.json()

    # Try to retrieve from Redis cache.
    cache_key = json.dumps(board)
    cache_key = cache_key.encode('utf-8')
    cached_prediction = self.redis_store.get(cache_key)
    if cached_prediction:
      cached_prediction = numpy.frombuffer(cached_prediction, dtype=numpy.float32)
      cached_prediction = cached_prediction.reshape(1, 1)
      print("FROM CACHE")
      return cached_prediction

    prediction = self.oracle.predict([board], verbose=0)

    # Commit to Redis cache.
    cached_prediction = prediction.tobytes()
    self.redis_store.set(cache_key, cached_prediction)

    print("FROM COMPUTATION")
    return prediction

oracle_path = "/home/dimitri/code/othello/output/20230828091759/models/eOthello-1/100.keras"
app = OracleDeployment.bind(oracle_path=oracle_path)

# Deploy the application locally.
ray.serve.run(app)
ray.serve.run(OracleDeployment.options(num_replicas=8).bind(oracle_path=oracle_path))

# This will keep the script running indefinitely
try:
  while True:
    time.sleep(1)
except KeyboardInterrupt:
  print("Shutting down...")
