import _othello_environment
import config
import json
import os
import pandas
import rules.othello

num_games_for_supervised_training = _othello_environment.parameter('OTHELLO_NUM_GAMES_FOR_SUPERVISED_TRAINING')
output_path = _othello_environment.parameter('OTHELLO_OUTPUT_PATH')

def decode_moves(moves):
  # split into pairs
  move_pairs = [moves[i:i+2].upper() for i in range(0, len(moves), 2)]

  def decode_move(move):
    letter, number = move
    x = ord(letter) - ord('A')  # 'a' -> 1, 'b' -> 2, ..., 'h' -> 8
    y = int(number) - 1 # '1' -> 1, '2' -> 2, ..., '8' -> 8
    return (y, x)

  tuples = [(move_number + 1, move_alpha, decode_move(move_alpha)) for move_number, move_alpha in enumerate(move_pairs)]
  return tuples

games = []
with open('data/othello_dataset.csv') as othello_dataset_file:
  othello_dataset_file.readline()
  i = 0
  for line in othello_dataset_file:
    i += 1
    if i > num_games_for_supervised_training: break
    # Remove the newline character at the end
    line = line.strip()
    line = line.split(',')
    game = {
      'name': line[0],
      'black_outcome': int(line[1]),
      'moves': line[2],
    }
    games.append(game)

othello_actions = {
  'Game': [],
  'Step': [],
  'Player': [],
  'Move': [],
  'Board': [],
  'Real_Player_Outcome': [],
}

for game in games:

  # 1: Black
  # 0: Draw
  # -1: White
  game_name = game['name']
  game_outcome = game['black_outcome']
  moves = game['moves']

  game = rules.othello.Othello()
  game.initialize_board()
  game.current_player = 0
  decoded_moves = decode_moves(moves)

  for move_number, move_alpha, move in decoded_moves:

    # initialize the new lists with zeros
    black = [[0]*len(row) for row in game.board]
    white = [[0]*len(row) for row in game.board]

    # populate the new lists
    for row_index in range(len(game.board)):
      for col_index in range(len(game.board[row_index])):
        if game.board[row_index][col_index] == 1:
          black[row_index][col_index] = 1
        elif game.board[row_index][col_index] == 2:
          white[row_index][col_index] = 1

    current_player_name = None
    if game.current_player == 0:
      current_player_name = 'Black'
      real_player_outcome = game_outcome
      neural_network_input = [black, white]
    if game.current_player == 1:
      current_player_name = 'White'
      real_player_outcome = -game_outcome
      neural_network_input = [white, black]
    if current_player_name is None:
      raise ValueError('current_player_name is None')

    # Save this learnable piece of wisdom, containing an observation and an outcome
    othello_actions['Game'].append(game_name)
    othello_actions['Step'].append(move_number)
    othello_actions['Player'].append(current_player_name)
    othello_actions['Move'].append(move_alpha)
    othello_actions['Board'].append(neural_network_input)
    othello_actions['Real_Player_Outcome'].append(real_player_outcome)

    # Prepare the next board
    game.move = move
    game.make_move()
    game.current_player = 1 - game.current_player
    if not game.has_legal_move():
      game.current_player = 1 - game.current_player

othello_actions_dataframe = pandas.DataFrame(othello_actions)
othello_actions_dataframe.set_index(['Game', 'Step'], inplace=True)

os.makedirs(output_path, exist_ok=True)
othello_actions_dataframe.to_csv(f'{output_path}/othello_prediction_prompts.csv', sep=';')

# Data to be written to the JSON file
data = {
  'num_games_for_supervised_training': num_games_for_supervised_training,
  'num_states': len(othello_actions_dataframe),
}

# Open a file for writing and save the data as JSON
with open(f'{output_path}/parameters.json', 'w') as json_file:
  json.dump(data, json_file, indent=2)
