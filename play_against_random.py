from info import info
import csv
import datetime
import multiprocessing
import os
import pandas
import random
import rules.othello
import sys

def get_env_variable(name):
  value = os.environ.get(name)
  if value is None:
    print(f"Error: Environment variable {name} not set.")
    sys.exit(1)
  return value

output_path = get_env_variable('OTHELLO_OUTPUT_PATH')
number_of_games = int(get_env_variable('OTHELLO_NUMBER_OF_GAMES'))



def index_to_letter(index):
  return chr(index + ord('a')).upper()

def index_to_number(index):
  return index + 1

def convert_index_to_chess_notation(index_tuple):
  return index_to_letter(index_tuple[1]) + str(index_to_number(index_tuple[0]))

def play_games(model_load_path, num_plays_for_evaluation):
  play_save_path = f'{model_load_path.replace("models", "plays", 1)}.csv'

  with open(play_save_path, 'w') as random_evaluation_file:
    print('evaluation_game_id,model_load_path,baseline,black_outcome,game_moves', file=random_evaluation_file)

  for _ in range(num_plays_for_evaluation):
    game = rules.othello.Othello(should_draw_tiles=False, model_path=model_load_path)
    game.draw_board()
    game.initialize_board()
    game.current_player = 0
    other_player_has_a_move = True
    moves = []
    black_outcome = None
    random_integer = random.randint(0, 100_000)
    random_integer_str = str(random_integer).zfill(5)
    game_id = f'{random_integer_str}'
    initial_state_root = game.root
    while True:
      # Check if the player pointed at by `game.current_player` has a legal move.
      # So, if there is a legal move from the current game state (board + player).
      if game.has_legal_move():
        other_player_has_a_move = True
        move = None
        if game.current_player == 0: # == BLACK PLAYER
          # BLACK is random
          move = game.make_random_move()
          game.root = game.root.next_node(move, game.current_player)
        if game.current_player == 1: # == WHITE PLAYER
          # WHITE is trained
          move = game.make_move_with_mcts(game.root)
          game.root = game.root.next_node(move, game.current_player)
        assert move is not None
        print(move)
        moves.append(move)
      else:
        if other_player_has_a_move is False:
          if game.num_tiles[0] > game.num_tiles[1]:
            black_outcome = 1
          elif game.num_tiles[0] < game.num_tiles[1]:
            black_outcome = -1
          else:
            black_outcome = 0
          break
        other_player_has_a_move = False
      game.current_player = 1 - game.current_player
      if not game.has_legal_move():
        game.current_player = 1 - game.current_player

    now = datetime.datetime.now()
    timestamp = now.strftime("output_%Y%m%d%H%M%S")
    with open(f'{output_path}/mcts/{timestamp}.dot', 'w') as file:
        file.write(initial_state_root.to_dot())

    assert black_outcome is not None
    translated_moves = [convert_index_to_chess_notation(move) for move in moves]
    moves_concatenation = ''.join(translated_moves).lower()
    row = f'{game_id},{model_load_path},RANDOM_PLAYER,{black_outcome},{moves_concatenation}'
    with open(play_save_path, 'a') as random_evaluation_file:
      print(row, file=random_evaluation_file)
    print(f'Written row to {play_save_path}.')
  return play_save_path

# Update CSV in a thread-safe manner
def update_csv(filename, data):
  with multiprocessing.Lock():
    with open(filename, 'a', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(data)

def worker(job_queue, csv_filename):
  while True:
    job = job_queue.get()
    if job is None:
      break

    play_save_path = play_games(job, number_of_games)
    a = pandas.read_csv(play_save_path)
    number_of_wins = a[a['black_outcome'] == -1].shape[0]
    assert number_of_games == a.shape[0]
    update_csv(csv_filename, [play_save_path, number_of_wins, number_of_games])

import os
def get_files_from_directory(directory):
  return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def main():
  random.seed(1)
  models_directory = f'{output_path}/models/eOthello-1/'
  model_load_paths = get_files_from_directory(models_directory)
  model_load_paths.sort()
  play_directory = models_directory.replace('models', 'plays', 1)
  os.makedirs(play_directory, exist_ok=True)
  os.makedirs(f'{output_path}/mcts', exist_ok=True)

  directory_path = f'{output_path}/models/eOthello-1/'
  model_load_paths = get_files_from_directory(directory_path)
  num_processes = 1

  job_queue = multiprocessing.Queue()
  for job in model_load_paths:
    job_queue.put(job)

  # Using sentinel values to tell processes when to exit
  for _ in range(num_processes):
    job_queue.put(None)

  csv_filename = f'{output_path}/winning_rates.csv'
  # Initialize the CSV with headers
  with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['play_path', 'number_of_wins', 'number_of_games'])

  # draw yellow horizontal baseline in stories.py into the diagram
  # play_games_against_random(number_of_games)

  processes = []
  for _ in range(num_processes):
    p = multiprocessing.Process(target=worker, args=(job_queue, csv_filename))
    processes.append(p)
    p.start()

  for p in processes:
    p.join()

if __name__ == "__main__":
  main()
