from info import info
import ast
import json
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

# Use ast.literal_eval to convert strings to lists, then convert lists to numpy arrays
boards = othello_actions_dataframe['Board']
boards = boards.apply(ast.literal_eval)
boards = boards.apply(numpy.array)
boards = boards.tolist()
boards = numpy.array(boards)
batch_size = 64
epochs = 1_000
num_epochs_per_checkpoint = 100
filepath = "generated/models/eOthello/eOthello-{epoch}.keras"

# Save model every 100 epochs
class SaveModelsCallback(tensorflow.keras.callbacks.Callback):
  def __init__(self):
    super(SaveModelsCallback, self).__init__()

  # Here, the model itself is before the epoch's updates.
  # If you evaluate this model, you get the loss of the weights before this epoch's updates.
  # The logs do not contain the loss at all.
  def on_epoch_begin(self, epoch, logs=None):
    if epoch % num_epochs_per_checkpoint == 0:
      self.model.save(filepath.format(epoch=epoch))

  # Here, the model itself is after the epoch's updates.
  # If you evaluate this model, you get the loss of the weights after this epoch's updates.
  # WARNING: The logs, however, contain the loss of the weights BEFORE the update.
  # WARNING: THIS IS NOT EXPLICITLY DOCUMENTED ANYWHERE!
  # This has been tested by comparing the loss in logs
  # against the loss when the model was evaluated in 'on_epoch_begin'.
  def on_epoch_end(self, epoch, logs=None):
    if epoch == epochs - 1:
      after_last_epoch = epoch + 1
      readable_epoch = f'{after_last_epoch:_}'
      self.model.save(filepath.format(epoch=readable_epoch))

# Prepare callback to save model every 100 epochs
checkpoint_callback = SaveModelsCallback()

# WARNING: The history object from model.fit() contains losses starting from the loss
# BEFORE the first epoch, then it progresses until the loss before the last epoch
# The loss after the last epoch has to be computed separately.
# Also, if you get the loss from logs['loss'] in the callback method on_epoch_end,
# then you will also get the loss before the current epoch, not the loss after.
# In this case, the documentation at
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback#on_epoch_end
# is misleading.

# Train the model with the new callback
history = model.fit(
  boards,
  othello_actions_dataframe['Real_Player_Outcome'],
  batch_size=batch_size,
  epochs=epochs,
  callbacks=[checkpoint_callback]
)

final_loss, final_accuracy = model.evaluate(
  boards,
  othello_actions_dataframe['Real_Player_Outcome'],
  verbose=0
)

# Save the loss dataframe to a CSV file
history_df = pandas.DataFrame(history.history)
history_df.reset_index(inplace=True)
history_df.rename(columns={'index': 'epoch'}, inplace=True)

# Create a DataFrame for initial metrics and append it to the history DataFrame
final_metrics_df = pandas.DataFrame({
    'epoch': [epochs],
    'loss': [final_loss],
    'mean_absolute_error': [final_accuracy]
})
history_df = pandas.concat([history_df, final_metrics_df])

history_df = history_df.iloc[::-1]

# Save the dataframe to a CSV file
history_df.to_csv('generated/training_history.csv', index=False)

# Load the existing data from the JSON file
with open('generated/parameters.json', 'r') as json_file:
    data = json.load(json_file)

# Add the new key-value pair
data["training_epochs"] = epochs
data["training_batch_size_per_step"] = batch_size
data["num_epochs_per_checkpoint"] = num_epochs_per_checkpoint
data["num_checkpoints"] = epochs // num_epochs_per_checkpoint + 1

# Save the updated data back to the JSON file
with open('generated/parameters.json', 'w') as json_file:
    json.dump(data, json_file, indent=2)
