from info import info
import os
import random
import rules.othello

def get_filepaths(directory):
  filepaths = []
  for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
      filepaths.append(os.path.join(dirpath, filename))
  return filepaths

def index_to_letter(index):
  return chr(index + ord('a')).upper()

def index_to_number(index):
  return index + 1

def convert_index_to_chess_notation(index_tuple):
  return index_to_letter(index_tuple[1]) + str(index_to_number(index_tuple[0]))

if __name__ == '__main__':

  random.seed(1)

  models_directory = 'generated/models/eOthello/'
  model_load_paths = get_filepaths(models_directory)
  model_load_paths.sort()
  play_directory = models_directory.replace('models', 'plays', 1)
  os.makedirs(play_directory, exist_ok=True)
  num_plays_for_evaluation = 20
  for model_load_path in model_load_paths:
    play_save_path = model_load_path.replace('models', 'plays', 1)
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
            game.report_result()
            break
          other_player_has_a_move = False
        game.current_player = 1 - game.current_player

      assert black_outcome is not None
      translated_moves = [convert_index_to_chess_notation(move) for move in moves]
      moves_concatenation = ''.join(translated_moves).lower()
      row = f'{game_id},{model_load_path},RANDOM_PLAYER,{black_outcome},{moves_concatenation}'
      random_evaluation_path = f'{play_save_path}.csv'
      if not os.path.exists(random_evaluation_path):
        with open(random_evaluation_path, 'w') as random_evaluation_file:
          print('evaluation_game_id,model_load_path,baseline,black_outcome,game_moves', file=random_evaluation_file)
      with open(random_evaluation_path, 'a') as random_evaluation_file:
        print(row, file=random_evaluation_file)
