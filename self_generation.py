import _othello_environment
import csv
import datetime
import graphviz
import mcts
import multiprocessing
import os
import pandas
import random
import rules.othello
import subprocess
import time
import variables_info

NUM_PROCESSES = _othello_environment.parameter('OTHELLO_NUM_PROCESSES')
NUMBER_OF_GAMES = _othello_environment.parameter('OTHELLO_NUMBER_OF_GAMES')
OUTPUT_PATH = _othello_environment.parameter('OTHELLO_OUTPUT_PATH')

def index_to_letter(index):
  return chr(index + ord('a')).upper()

def index_to_number(index):
  return index + 1

def convert_index_to_chess_notation(index_tuple):
  return index_to_letter(index_tuple[1]) + str(index_to_number(index_tuple[0]))

def play_game(model_load_path):
  game = rules.othello.Othello(should_draw_tiles=False)
  game.draw_board()
  game.initialize_board()
  game.current_player = 0
  other_player_has_a_move = True
  moves = []
  black_outcome = None
  initial_state_root = mcts.Node(model_load_path, game)
  initial_state_root.expand()
  current_root = initial_state_root
  while True:
    # Check if the player pointed at by `game.current_player` has a legal move.
    # So, if there is a legal move from the current game state (board + player).
    if game.has_legal_move():
      other_player_has_a_move = True
      move = game.make_exploratory_move_with_mcts(current_root)
      current_root = current_root.next_node(move, game.current_player)
      assert move is not None
      print(move)
      moves.append(move)
    else:
      if other_player_has_a_move is False:
        black_outcome = game.get_black_outcome()
        break
      other_player_has_a_move = False
    game.current_player = 1 - game.current_player
    if not game.has_legal_move():
      game.current_player = 1 - game.current_player

  for _ in range(5):
    with multiprocessing.Lock():
      # Loading the .dot string into Graphviz.
      # See: https://graphviz.readthedocs.io/en/stable/examples.html
      dot_string = initial_state_root.to_dot()
      graphviz_source = graphviz.Source(dot_string)
      graphviz_source.format = 'svg'

      now = datetime.datetime.now()
      timestamp = now.strftime('%Y%m%d%H%M%S')
      dot_file_path = f'{OUTPUT_PATH}/mcts/{timestamp}.dot'
      json_file_path = f'{OUTPUT_PATH}/mcts/{timestamp}.json'

      # This render() function will write TWO files:
      # First, it writes the DOT file to the given path.
      # Second, it writes the SVG file to the given path, followed by '.svg'.
      try:
        initial_state_root.to_json(json_file_path)
        break
      except subprocess.CalledProcessError:
        # Sometimes, we get:
        # subprocess.CalledProcessError: Command '[PosixPath('dot'), '-Kdot', '-Tsvg', '-O', '20230824090520.dot']' returned non-zero exit status 1.
        # But the .svg file is created and is well-readable. So, we ignore all called process errors.
        time.sleep(1)
        pass

  assert black_outcome is not None
  return black_outcome, moves, timestamp

def worker(job_queue):
  while True:
    model_load_path = job_queue.get()
    if model_load_path is None:
      return

    variables_info.d(model_load_path)

    # The ", 1" is there to protect the model's name to not be changed (in case, the name contains "models").
    play_path = model_load_path.replace('models', 'selfplay', 1)
    play_path = f'{play_path}.csv'

    # If the file does not exist yet, create one and put the headers in there.
    if not os.path.exists(play_path):
      with open(play_path, 'w') as random_evaluation_file:
        print('evaluation_game_id,model_load_path,baseline,black_outcome,game_moves', file=random_evaluation_file)

    black_outcome, moves, timestamp = play_game(model_load_path)

    translated_moves = [convert_index_to_chess_notation(move) for move in moves]
    moves_concatenation = ''.join(translated_moves).lower()

    row = f'{timestamp},{model_load_path},RANDOM_PLAYER,{black_outcome},{moves_concatenation}'
    with open(play_path, 'a') as random_evaluation_file:
      print(row, file=random_evaluation_file)
      print(f'Written row to {play_path}.')

def get_files_from_directory(directory):
  return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

if __name__ == '__main__':
  random.seed(1)
  models_directory = f'{OUTPUT_PATH}/models/eOthello-1'
  model_load_paths = ['/content/othello/output/20230828091759/models/eOthello-11/10_000']
  model_load_paths.sort()
  play_directory = models_directory.replace('models', 'selfplay', 1)
  os.makedirs(play_directory, exist_ok=True)
  os.makedirs(f'{OUTPUT_PATH}/mcts', exist_ok=True)

  job_queue = multiprocessing.Queue()
  same_model_for_many_games = model_load_paths * NUMBER_OF_GAMES
  same_model_for_many_games.sort()
  for model_load_path in same_model_for_many_games:
    job_queue.put(model_load_path)

  # Using sentinel values to tell processes when to exit
  for _ in range(NUM_PROCESSES):
    job_queue.put(None)

  processes = []
  for _ in range(NUM_PROCESSES):
    p = multiprocessing.Process(target=worker, args=(job_queue,))
    processes.append(p)
    p.start()

  for p in processes:
    p.join()

  winning_rates_csv_path = f'{OUTPUT_PATH}/winning_rates.csv'
  with open(winning_rates_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['play_path', 'number_of_wins', 'number_of_games'])

  play_paths = get_files_from_directory(play_directory)
  for play_path in play_paths:
    plays_table = pandas.read_csv(play_path)
    # The oracle-led player controls White.
    number_of_wins = plays_table[plays_table['black_outcome'] == -1].shape[0]
    with open(winning_rates_csv_path, 'a', newline='') as csvfile:
      writer = csv.writer(csvfile)
      # actual_number_of_games might differ from NUMBER_OF_GAMES,
      # if the plays directory wasn't empty, when the script ran.
      actual_number_of_games = plays_table.shape[0]
      writer.writerow([play_path, number_of_wins, actual_number_of_games])
