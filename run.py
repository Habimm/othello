from info import info
import numpy
import pandas
import random
import tensorflow
import tensorflow.keras
import tensorflow.keras.layers

# Load the decomposed games data.
othello_actions_dataframe = pandas.read_csv('data/othello_prediction_prompts.csv', sep=';', index_col=['Game', 'Step'])
info(othello_actions_dataframe)
info(othello_actions_dataframe['Board'])

# Fix the sequence of random numbers.
RANDOM_NUMBERS_SEQUENCE_NAME = 1
numpy.random.seed(RANDOM_NUMBERS_SEQUENCE_NAME)
random.seed(RANDOM_NUMBERS_SEQUENCE_NAME)
tensorflow.random.set_seed(RANDOM_NUMBERS_SEQUENCE_NAME)

# Define the oracle neural network's architecture
inputs = tensorflow.keras.Input(shape=(2, 8, 8)) # input shape is (8, 8, 2) - two 8x8 planes
x = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(inputs)
x = tensorflow.keras.layers.Flatten()(x) # Flatten the tensor for the Dense layer
outputs = tensorflow.keras.layers.Dense(1, activation='tanh')(x)
model = tensorflow.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tensorflow.keras.metrics.MeanAbsoluteError()])
batch_size = 64

# Convert 'Board' column to 3D numpy array
import numpy as np
import ast

# Use ast.literal_eval to convert strings to lists, then convert lists to numpy arrays
boards = np.array(othello_actions_dataframe['Board'].apply(ast.literal_eval).apply(np.array).tolist())

# Print information about the numpy arrays
print(boards.shape)
print(boards.dtype)
print(type(boards[0]))
info(boards)
info(boards[0])
info(othello_actions_dataframe['Real_Player_Outcome'].tolist())
breakpoint()

history = model.fit(othello_actions_dataframe['Board'], othello_actions_dataframe['Real_Player_Outcome'], batch_size=batch_size, epochs=1000)
othello_actions_dataframe['Predicted_Player_Outcome'] = model.predict(othello_actions_dataframe['Board'])

# Save some training statistics to CSV file.
hist_df = pandas.DataFrame(history.history)
hist_df.reset_index(inplace=True)
hist_df.rename(columns={'index': 'Epoch'}, inplace=True)
hist_df['Epoch'] += 1
hist_df = hist_df.iloc[::-1]
hist_df.to_csv('training_history.csv', index=False)

othello_actions_dataframe.to_csv('data/othello_predictions.csv', sep=';')
