from info import info
import os
import random
import rules.othello

def index_to_letter(index):
  return chr(index + ord('a')).upper()

def index_to_number(index):
  return index + 1

def convert_index_to_chess_notation(index_tuple):
  # Inverting the indices because of the different convention
  return index_to_letter(index_tuple[1]) + str(index_to_number(index_tuple[0]))

if __name__ == '__main__':
  random.seed(1)
  # model_load_path = 'generated/othello_model.keras-1000.keras-1000.keras'
  # model_load_path = 'generated/othello_model.keras-1000.keras'
  # model_load_path = 'generated/othello_model.keras'

  # model_load_path = 'generated/othello_model.keras-200.keras'
  model_load_path = 'generated/othello_model.keras-1000.keras'

  for _ in range(100):
    game = rules.othello.Othello(should_draw_tiles=False, model_path=model_load_path)
    game.draw_board()
    game.initialize_board()

    # Starts playing the game
    # The user makes a move by clicking one of the squares on the board
    # The computer makes a random legal move every time
    # Game is over when there are no more lagal moves or the board is full

    # game.run()

    # 0 = BLACK PLAYER
    # 1 = WHITE PLAYER
    game.current_player = 0
    other_player_has_a_move = True
    moves = []
    black_outcome = None
    random_integer = random.randint(0, 100_000)
    random_integer_str = str(random_integer).zfill(5)
    game_id = f'RANDOM_{random_integer_str}'
    while True:
      # Check if the player pointed at by `game.current_player` has a legal move.
      # So, if there is a legal move from the current game state (board + player).
      if game.has_legal_move():
        other_player_has_a_move = True
        move = None
        if game.current_player == 0:
          # BLACK is random
          move = game.make_random_move()
        if game.current_player == 1:
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
    info(moves)
    print(convert_index_to_chess_notation((2, 4)))
    translated_moves = [convert_index_to_chess_notation(move) for move in moves]
    moves_concatenation = ''.join(translated_moves).lower()
    info(moves_concatenation)

    row = f'{game_id},{black_outcome},{moves_concatenation}'

    random_evaluation_path = f'{model_load_path}-random_evaluation.csv'

    if not os.path.exists(random_evaluation_path):
      with open(random_evaluation_path, 'w') as random_evaluation_file:
        print('SelfPlay_game_id,black_outcome,game_moves', file=random_evaluation_file)

    with open(random_evaluation_path, 'a') as random_evaluation_file:
      print(row, file=random_evaluation_file)
