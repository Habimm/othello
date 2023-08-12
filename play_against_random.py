from info import info
import csv
import multiprocessing
import os
import os
import pandas
import random
import rules.othello

number_of_games = 20

def index_to_letter(index):
  return chr(index + ord('a')).upper()

def index_to_number(index):
  return index + 1

def convert_index_to_chess_notation(index_tuple):
  return index_to_letter(index_tuple[1]) + str(index_to_number(index_tuple[0]))

def play_games(model_load_path, num_plays_for_evaluation):
  play_save_path = f'{model_load_path.replace("models", "plays", 1)}.csv'
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
    while True:
      # Check if the player pointed at by `game.current_player` has a legal move.
      # So, if there is a legal move from the current game state (board + player).
      if game.has_legal_move():
        other_player_has_a_move = True
        move = None
        if game.current_player == 0: # == BLACK PLAYER
          # BLACK is random
          move = game.make_random_move()
        if game.current_player == 1: # == WHITE PLAYER
          # WHITE is trained
          move = game.make_move_with_best_value()
        assert move is not None
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

    assert black_outcome is not None
    translated_moves = [convert_index_to_chess_notation(move) for move in moves]
    moves_concatenation = ''.join(translated_moves).lower()
    row = f'{game_id},{model_load_path},RANDOM_PLAYER,{black_outcome},{moves_concatenation}'
    if not os.path.exists(play_save_path):
      with open(play_save_path, 'w') as random_evaluation_file:
        print('evaluation_game_id,model_load_path,baseline,black_outcome,game_moves', file=random_evaluation_file)
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
  models_directory = 'generated_200_plays/models/eOthello/'
  model_load_paths = get_files_from_directory(models_directory)
  model_load_paths.sort()
  play_directory = models_directory.replace('models', 'plays', 1)
  os.makedirs(play_directory, exist_ok=True)

  directory_path = 'generated_200_plays/models/eOthello/'
  model_load_paths = get_files_from_directory(directory_path)
  num_processes = min(multiprocessing.cpu_count(), len(model_load_paths))

  job_queue = multiprocessing.Queue()
  for job in model_load_paths:
    job_queue.put(job)

  # Using sentinel values to tell processes when to exit
  for _ in range(num_processes):
    job_queue.put(None)

  csv_filename = 'generated_200_plays/winning_rates.csv'
  # Initialize the CSV with headers
  with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['play_path', 'number_of_wins', 'number_of_games'])

  processes = []
  for _ in range(num_processes):
    p = multiprocessing.Process(target=worker, args=(job_queue, csv_filename))
    processes.append(p)
    p.start()

  for p in processes:
    p.join()

if __name__ == "__main__":
  main()
