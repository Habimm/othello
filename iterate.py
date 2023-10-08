import _othello_environment
import _train
import csv
import datetime
import decompose_games
import graphviz
import mcts
import multiprocessing
import oracle
import os
import pandas
import random
import rules.othello
import self_generation
import shutil
import subprocess
import time
import variables_info

OUTPUT_PATH = _othello_environment.parameter('OTHELLO_OUTPUT_PATH')

os.makedirs(f'{OUTPUT_PATH}/0/plays', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/0/models', exist_ok=True)

decompose_games.decompose(
  source_path = f'data/othello_dataset.csv',
  target_path = f'{OUTPUT_PATH}/0/plays/states.csv',
  trajectories_path = f'{OUTPUT_PATH}/0/plays/trajectories.csv',
)

last_filepath = _train.train(
  prediction_prompts_path = f'{OUTPUT_PATH}/0/plays/states.csv',
  models_path = f'{OUTPUT_PATH}/0/models',
)
