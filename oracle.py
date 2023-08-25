from info import info
import ray
import ray.serve
import starlette.requests
import tensorflow
import time
import typing

@ray.serve.deployment(route_prefix="/predict")
class OracleDeployment:
  def __init__(self, oracle_path: str):
    self.oracle = tensorflow.keras.models.load_model(oracle_path)

  async def __call__(self, request: starlette.requests.Request) -> typing.Dict:
    board = await request.json()
    prediction = self.oracle.predict([board], verbose=0)
    return prediction

oracle_path = "/home/dimitri/code/othello/output/generated_20230814012642_mcts/models/eOthello-1/1_000.keras"
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
