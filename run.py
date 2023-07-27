from info import info
import numpy
import othello_game.othello
import pandas
import random
import tensorflow
import tensorflow.keras
import tensorflow.tensorflow.keras.layers

# Fix the sequence of random numbers.
RANDOM_NUMBERS_SEQUENCE_NAME = 1
numpy.random.seed(RANDOM_NUMBERS_SEQUENCE_NAME)
random.seed(RANDOM_NUMBERS_SEQUENCE_NAME)
tensorflow.random.set_seed(RANDOM_NUMBERS_SEQUENCE_NAME)

# Define the oracle neural network's architecture
inputs = tensorflow.keras.Input(shape=(2, 8, 8)) # input shape is (8, 8, 2) - two 8x8 planes
x = tensorflow.tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(inputs)
x = tensorflow.tensorflow.keras.layers.Flatten()(x) # Flatten the tensor for the Dense layer
outputs = tensorflow.tensorflow.keras.layers.Dense(1, activation='tanh')(x)
model = tensorflow.keras.Model(inputs=inputs, outputs=outputs)
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

games = []
with open('data/othello_dataset.csv') as othello_dataset_file:
  othello_dataset_file.readline()
  i = 0
  for line in othello_dataset_file:
    i += 1
    if i > 200: break
    # Remove the newline character at the end
    line = line.strip()
    line = line.split(',')
    game = {
      "name": line[0],
      "black_outcome": int(line[1]),
      "moves": line[2],
    }
    games.append(game)

othello_actions = {
  "Game": [],
  "Step": [],
  "Player": [],
  "Move": [],
  "Board": [],
  "Real_Player_Outcome": [],
}

for game in games:

  # 1: Black
  # 0: Draw
  # -1: White
  game_name = game["name"]
  game_outcome = game["black_outcome"]
  moves = game["moves"]

  game = othello_game.othello.Othello()
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
      raise ValueError("current_player_name is None")

    # Save this learnable piece of wisdom, containing an observation and an outcome
    othello_actions["Game"].append(game_name)
    othello_actions["Step"].append(move_number)
    othello_actions["Player"].append(current_player_name)
    othello_actions["Move"].append(move_alpha)
    othello_actions["Board"].append(neural_network_input)
    othello_actions["Real_Player_Outcome"].append(real_player_outcome)

    # Prepare the next board
    game.move = move
    game.make_move()
    game.current_player = 1 - game.current_player
    if not game.has_legal_move():
      game.current_player = 1 - game.current_player



othello_actions_dataframe = pandas.DataFrame(othello_actions)
othello_actions_dataframe.set_index(["Game", "Step"], inplace=True)
othello_actions_dataframe.to_csv("othello_actions_dataframe.csv", sep=";")
info(othello_actions_dataframe)

othello_actions_dataframe_loaded = pandas.read_csv("othello_actions_dataframe.csv", sep=";", index_col=["Game", "Step"])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tensorflow.keras.metrics.MeanAbsoluteError()])
batch_size = 64
history = model.fit(othello_actions["Board"], othello_actions["Real_Player_Outcome"], batch_size=batch_size, epochs=1000)
othello_actions_dataframe_loaded['Predicted_Player_Outcome'] = model.predict(othello_actions["Board"])

# Convert the history.history dict to a pandas DataFrame
hist_df = pandas.DataFrame(history.history)

# Reset the DataFrame index to add iteration (epoch) numbers starting from 1
hist_df.reset_index(inplace=True)

# Rename the 'index' column to 'Epoch' (Epoch numbers start at 1, not 0)
hist_df.rename(columns={'index': 'Epoch'}, inplace=True)
hist_df['Epoch'] += 1

hist_df = hist_df.iloc[::-1]

# Save it to csv
hist_df.to_csv('training_history.csv', index=False)

othello_actions_dataframe_loaded.to_csv("othello_actions_dataframe_with_pred.csv", sep=";")
