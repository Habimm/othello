from info import info
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import othello_game.othello
import random
import tensorflow as tf

# Fix the sequence of random numbers.
RANDOM_NUMBERS_SEQUENCE_NAME = 1
np.random.seed(RANDOM_NUMBERS_SEQUENCE_NAME)
random.seed(RANDOM_NUMBERS_SEQUENCE_NAME)
tf.random.set_seed(RANDOM_NUMBERS_SEQUENCE_NAME)

# Define the oracle neural network's architecture
inputs = keras.Input(shape=(2, 8, 8)) # input shape is (8, 8, 2) - two 8x8 planes
x = layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(inputs)
x = layers.Flatten()(x) # Flatten the tensor for the Dense layer
outputs = layers.Dense(1, activation='tanh')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

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

# 1: Black
# 0: Draw
# -1: White
game_name = '1050515'
game_outcome = -1
moves = 'd3c5f6f5e6e3d6f7b6d7e2c6d8c4e7c3f4c8c7d2d1a6b3f3b8f8b5f2e8a8b4b7a7a5g3f1g4g5h6a4g6e1b2c1c2h3h4g2g7h2h1a1g1b1a2a3h7h8g8h5'

games = []
with open('data/othello_dataset.csv') as dataset_file:
  dataset_file.readline()
  i = 0
  for line in dataset_file:
    i += 1
    if i > 2: break
    line = line.split(',')
    game = {
      "name": line[0],
      "black_outcome": int(line[1]),
      "moves": line[2],
    }
    games.append(game)

info(games)
breakpoint()

game = othello_game.othello.Othello()
game.initialize_board()
game.current_player = 0
decoded_moves = decode_moves(moves)

all_othello_positions = []
real_player_outcomes = []
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
    raise ValueError("current_player_name is None")

  all_othello_positions.append(neural_network_input)
  real_player_outcomes.append(real_player_outcome)
  game.move = move
  game.make_move()
  game.current_player = 1 - game.current_player
  if not game.has_legal_move():
    game.current_player = 1 - game.current_player


model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.MeanAbsoluteError()])
model.fit(np.array(all_othello_positions), np.array(real_player_outcomes), epochs=1000)
test_loss, test_acc = model.evaluate(all_othello_positions, real_player_outcomes)
all_predicted_player_outcomes = model.predict(all_othello_positions)

# ===================================================================================================================

game = othello_game.othello.Othello()
game.initialize_board()
game.current_player = 0
decoded_moves = decode_moves(moves)

print(f'Game, Step, Player, Move, Board, Real_Player_Outcome, Predicted_Player_Outcome')
for (move_number, move_alpha, move), predicted_player_outcome in zip(decoded_moves, all_predicted_player_outcomes):

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
    raise ValueError("current_player_name is None")

  print(f'{game_name}, {move_number}, {current_player_name}, {move_alpha}, {neural_network_input}, {real_player_outcome}, {predicted_player_outcome}')
  game.move = move
  game.make_move()
  game.current_player = 1 - game.current_player
  if not game.has_legal_move():
    game.current_player = 1 - game.current_player
