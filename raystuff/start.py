from info import info
import ray.serve
import starlette.requests
import tensorflow
import time
import typing

@ray.serve.deployment(route_prefix="/predict")
class MCTSDeployment:
  def __init__(self, mcts_oracle_path: str):
    self.mcts_oracle = tensorflow.keras.models.load_model(mcts_oracle_path)

  async def __call__(self, request: starlette.requests.Request) -> typing.Dict:
    board = await request.json()
    prediction = self.mcts_oracle.predict([board], verbose=0)
    return prediction

mcts_oracle_path = "/home/dimitri/code/othello/output/generated_20230814012642_mcts/models/eOthello-1/1_000.keras"
app = MCTSDeployment.bind(mcts_oracle_path=mcts_oracle_path)

# Deploy the application locally.
ray.serve.run(app)

# This will keep the script running indefinitely
try:
  while True:
    time.sleep(1)
except KeyboardInterrupt:
  print("Shutting down...")
