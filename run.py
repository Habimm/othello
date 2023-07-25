from info import info
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import tensorflow as tf

RANDOM_NUMBERS_SEQUENCE_NAME = 1

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(RANDOM_NUMBERS_SEQUENCE_NAME)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
random.seed(RANDOM_NUMBERS_SEQUENCE_NAME)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
tf.random.set_seed(RANDOM_NUMBERS_SEQUENCE_NAME)

# eOthello_game_id,winner,game_moves
# 1050515,-1,d3c5f6f5e6e3d6f7b6d7e2c6d8c4e7c3f4c8c7d2d1a6b3f3b8f8b5f2e8a8b4b7a7a5g3f1g4g5h6a4g6e1b2c1c2h3h4g2g7h2h1a1g1b1a2a3h7h8g8h5
# https://www.eothello.com/game/1050515

# Define the architecture
inputs = keras.Input(shape=(2, 8, 8)) # input shape is (8, 8, 2) - two 8x8 planes

# Add one hidden layer (I'm using a convolutional layer because the input is a 3D tensor)
x = layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(inputs)
x = layers.Flatten()(x) # Flatten the tensor for the Dense layer

# Add an output layer with a single neuron and sigmoid activation function (to get a number between 0 and 1)
outputs = layers.Dense(1, activation='tanh')(x)

# Instantiate the model
model = keras.Model(inputs=inputs, outputs=outputs)

# Print the model summary
model.summary()

# Game, Step, Player, Move, Player_Outcome, Board
# 1050515, 26, 1, F8, -1, [[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 2, 0, 0], [0, 0, 1, 1, 2, 2, 0, 0], [0, 0, 2, 2, 1, 2, 0, 0], [2, 2, 2, 1, 2, 2, 0, 0], [0, 0, 1, 1, 1, 2, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0]]

current = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 1, 0, 0],
  [0, 0, 1, 1, 0, 1, 0, 0],
  [1, 1, 1, 0, 1, 1, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
]

opponent = [
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 1, 1, 0, 0, 0],
  [0, 1, 1, 1, 1, 0, 0, 0],
  [0, 0, 1, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 1, 1, 1, 0, 0, 0],
  [0, 1, 1, 1, 0, 0, 0, 0],
]

example_input = [[current, opponent]]
info(example_input)

# Run the model on the example input
example_output = model.predict(example_input)

# Print the output
print(("Output:", example_output))
print(("Output:", example_output[0][0]))

moves = 'd3c5f6f5e6e3d6f7b6d7e2c6d8c4e7c3f4c8c7d2d1a6b3f3b8f8b5f2e8a8b4b7a7a5g3f1g4g5h6a4g6e1b2c1c2h3h4g2g7h2h1a1g1b1a2a3h7h8g8h5'

