import _othello_environment
import _train
import csv
import datetime
import decompose_games
import graphviz
import json
import mcts
import multiprocessing
import oracle
import os
import pandas
import random
import self_generation
import shutil
import subprocess
import time
import variables_info

OUTPUT_PATH = _othello_environment.parameter('OTHELLO_OUTPUT_PATH')

# ===================================================================================

last_filepath = None
for generation in range(6):

  os.makedirs(f'{OUTPUT_PATH}/{generation}/plays', exist_ok=True)
  os.makedirs(f'{OUTPUT_PATH}/{generation}/models', exist_ok=True)

  if generation >= 1:
    assert last_filepath is not None

    self_generation.self_generate(
      model_load_paths = [last_filepath],
      plays_path = f'{OUTPUT_PATH}/{generation}/plays',
    )
    data = {
      'model': last_filepath,
    }
    with open(f'{OUTPUT_PATH}/{generation}/plays/meta.json', 'w') as json_file:
      json.dump(data, json_file, indent=2)

  if generation == 0:
    decompose_games.decompose(
      source_path = f'data/othello_dataset.csv',
      target_path = f'{OUTPUT_PATH}/0/plays/states.csv',
      trajectories_path = f'{OUTPUT_PATH}/0/plays/trajectories.csv',
    )
  else:
    decompose_games.decompose(
      source_path = f'{OUTPUT_PATH}/{generation}/plays/trajectories.csv',
      target_path = f'{OUTPUT_PATH}/{generation}/plays/states.csv',
    )

  last_filepath = _train.train(
    prediction_prompts_path = f'{OUTPUT_PATH}/{generation}/plays/states.csv',
    models_path = f'{OUTPUT_PATH}/{generation}/models',
  )
