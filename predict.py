import ast
import numpy
import os
import pandas
import random
import tensorflow
import tensorflow.keras
import tensorflow.keras.callbacks
import tensorflow.keras.layers

# Load the decomposed games data.
othello_actions_dataframe = pandas.read_csv('generated/othello_prediction_prompts.csv', sep=';', index_col=['Game', 'Step'])

# Fix the sequence of random numbers.
RANDOM_NUMBERS_SEQUENCE_NAME = 1
numpy.random.seed(RANDOM_NUMBERS_SEQUENCE_NAME)
random.seed(RANDOM_NUMBERS_SEQUENCE_NAME)
tensorflow.random.set_seed(RANDOM_NUMBERS_SEQUENCE_NAME)

# model_load_path = 'generated/othello_model.keras-1000.keras-1000.keras'
# model_load_path = 'generated/othello_model.keras-1000.keras'
# model_load_path = 'generated/othello_model.keras'
model_load_path = None

# Check if the model load path exists
if not model_load_path:
  # Define the oracle neural network's architecture
  inputs = tensorflow.keras.Input(shape=(2, 8, 8)) # input shape is (8, 8, 2) - two 8x8 planes
  x = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(inputs)
  x = tensorflow.keras.layers.Flatten()(x) # Flatten the tensor for the Dense layer
  outputs = tensorflow.keras.layers.Dense(1, activation='tanh')(x)
  model = tensorflow.keras.Model(inputs=inputs, outputs=outputs)
  model.summary()
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tensorflow.keras.metrics.MeanAbsoluteError()])
else:
  model = tensorflow.keras.models.load_model(model_load_path)

os.makedirs("generated/models/eOthello", exist_ok=True)

# Save initial model before training begins
filepath = "generated/models/eOthello/eOthello-{epoch}.keras"
initial_save_path = filepath.replace('{epoch}', '0')
model.save(initial_save_path)

# Prepare callback to save model every 100 epochs
checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
  filepath, monitor='loss', verbose=1,
  save_best_only=False, save_weights_only=False,
  mode='auto', save_freq='epoch', options=None
)

# Use ast.literal_eval to convert strings to lists, then convert lists to numpy arrays
boards = othello_actions_dataframe['Board']
boards = boards.apply(ast.literal_eval)
boards = boards.apply(numpy.array)
boards = boards.tolist()
boards = numpy.array(boards)
batch_size = 64
epochs = 1_000

initial_loss, initial_accuracy = model.evaluate(
  boards,
  othello_actions_dataframe['Real_Player_Outcome'],
  verbose=0
)

# Train the model with the new callback
history = model.fit(
  boards,
  othello_actions_dataframe['Real_Player_Outcome'],
  batch_size=batch_size,
  epochs=epochs,
  callbacks=[checkpoint_callback]
)

# Save the loss dataframe to a CSV file
history_df = pandas.DataFrame(history.history)
history_df.reset_index(inplace=True)
history_df.rename(columns={'index': 'epoch'}, inplace=True)
history_df['epoch'] += 1

# Create a DataFrame for initial metrics and append it to the history DataFrame
initial_metrics_df = pandas.DataFrame({
    'epoch': [0],
    'loss': [initial_loss],
    'mean_absolute_error': [initial_accuracy]
})
history_df = pandas.concat([initial_metrics_df, history_df])

history_df = history_df.iloc[::-1]

# Save the dataframe to a CSV file
history_df.to_csv('generated/training_history.csv', index=False)
