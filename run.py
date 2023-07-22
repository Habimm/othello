from tensorflow import keras
from info import info
from tensorflow.keras import layers
import numpy as np

# Define the architecture
inputs = keras.Input(shape=(8, 8, 2)) # input shape is (8, 8, 2) - two 8x8 planes

# Add one hidden layer (I'm using a convolutional layer because the input is a 3D tensor)
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.Flatten()(x) # Flatten the tensor for the Dense layer

# Add an output layer with a single neuron and sigmoid activation function (to get a number between 0 and 1)
outputs = layers.Dense(1, activation='tanh')(x)

# Instantiate the model
model = keras.Model(inputs=inputs, outputs=outputs)

# Print the model summary
model.summary()

# Instantiate an example input
example_input = np.random.randint(0, 2, size=(1, 8, 8, 2))
info(example_input)

current = [
  [0, 0, 1, 0, 0, 0, 0, 0],
  [0, 0, 1, 1, 1, 0, 0, 0],
  [1, 0, 1, 1, 1, 1, 0, 0],
  [0, 1, 1, 0, 0, 1, 1, 0],
  [0, 0, 1, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 1, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
]

opponent = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 1, 1, 0, 0, 1],
  [0, 0, 0, 0, 1, 1, 1, 1],
  [0, 0, 0, 1, 0, 1, 1, 1],
  [0, 0, 1, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 0],
]

# Stack the arrays along a new third axis
example_input = np.stack((current, opponent), axis=-1) # shape will be (8, 8, 2)

# Expand dimensions to simulate a batch
example_input = np.expand_dims(example_input, axis=0) # shape will be (1, 8, 8, 2)

info(example_input)

# Run the model on the example input
example_output = model.predict(example_input)

# Print the output
print("Output:", example_output[0][0])
