import ast
import info
import comet_ml
import json
import numpy
import os
import pandas
import random
import sys
import tensorflow
import tensorflow.keras
import tensorflow.keras.callbacks
import tensorflow.keras.layers

# make ray ai for model deployment

experiment = comet_ml.Experiment(
    api_key="naHPlykXz5yy1LscJvnkLiyBJ",
    project_name="Othello"
)

# Metrics from this training run will now be
# available in the Comet UI

def get_env_variable(name):
    value = os.environ.get(name)
    if value is None:
        print(f"Error: Environment variable {name} not set.")
        sys.exit(1)
    return value

batch_size = int(get_env_variable('OTHELLO_BATCH_SIZE'))
epochs = int(get_env_variable('OTHELLO_EPOCHS'))
output_path = get_env_variable('OTHELLO_OUTPUT_PATH')
num_epochs_per_checkpoint = int(get_env_variable('OTHELLO_NUM_EPOCHS_PER_CHECKPOINT'))
random_seed = int(get_env_variable('OTHELLO_RANDOM_SEED'))



# Load the decomposed games data.
othello_actions_dataframe = pandas.read_csv(
  f'{output_path}/othello_prediction_prompts.csv',
  sep=';',
  index_col=['Game', 'Step'],
)

# Fix the sequence of random numbers.
numpy.random.seed(random_seed)
random.seed(random_seed)
tensorflow.random.set_seed(random_seed)

model_load_path = None
# model_load_path = f'{output_path}/models/eOthello-1/1_000.keras'
# Check if the model load path exists
if not model_load_path:
  # Define the oracle neural network's architecture
  inputs = tensorflow.keras.Input(shape=(8, 8, 2))
  x = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_last')(inputs)
  x = tensorflow.keras.layers.Flatten()(x)
  outputs = tensorflow.keras.layers.Dense(1, activation='tanh')(x)
  model = tensorflow.keras.Model(inputs=inputs, outputs=outputs)
  model.summary()
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tensorflow.keras.metrics.MeanAbsoluteError()])
  filepath = f'{output_path}/models/eOthello-1/{{epoch}}.keras'
else:
  model = tensorflow.keras.models.load_model(model_load_path)
  filepath = f'{model_load_path}-{{epoch}}.keras'

model_json_string = model.to_json()
model_config_dict = json.loads(model_json_string)

os.makedirs(f'{output_path}/models/eOthello-1', exist_ok=True)








# Save training parameter in a JSON file.
# Try to load the existing data from the JSON file
try:
  with open(f'{output_path}/parameters.json', 'r') as json_file:
    data = json.load(json_file)
except FileNotFoundError:
  # If the file doesn't exist, initialize data as an empty dictionary
  data = {}

# Add the new key-value pair
data['training_epochs'] = epochs
data['training_batch_size_per_step'] = batch_size
data['num_epochs_per_checkpoint'] = num_epochs_per_checkpoint
# + 1 because we also have the checkpoint at 0 epochs,
# before training starts
data['num_checkpoints'] = epochs // num_epochs_per_checkpoint + 1
data['model_config_dict'] = model_config_dict

# Save the updated data back to the JSON file
with open(f'{output_path}/parameters.json', 'w') as json_file:
  json.dump(data, json_file, indent=2)








# Use ast.literal_eval to convert strings to lists, then convert lists to numpy arrays
boards = othello_actions_dataframe['Board']
boards = boards.apply(ast.literal_eval)
boards = boards.apply(numpy.array)
boards = boards.tolist()
boards = numpy.array(boards)
boards_nhwc = tensorflow.transpose(boards, [0, 2, 3, 1])

# Save model every `num_epochs_per_checkpoint` epochs
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

# Prepare callback to save model every `num_epochs_per_checkpoint` epochs
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
  boards_nhwc,
  othello_actions_dataframe['Real_Player_Outcome'],
  batch_size=batch_size,
  epochs=epochs,
  callbacks=[checkpoint_callback]
)

final_loss, final_accuracy = model.evaluate(
  boards_nhwc,
  othello_actions_dataframe['Real_Player_Outcome'],
  verbose=0
)

# 1. Load the existing data from the CSV file
# Try to load the existing data from the CSV file
try:
    existing_df = pandas.read_csv(f'{output_path}/training_history.csv')
except FileNotFoundError:
    # If the file doesn't exist, create an empty DataFrame with the expected columns
    existing_df = pandas.DataFrame(columns=['epoch', 'loss', 'mean_absolute_error'])

# Save the loss dataframe to a new DataFrame
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

# 2. Concatenate the new data with the existing data
combined_df = pandas.concat([existing_df, history_df], ignore_index=True)

# Sort the data in reverse order (optional)
combined_df = combined_df.iloc[::-1]

# 3. Save the combined dataframe back to the CSV file
combined_df.to_csv(f'{output_path}/training_history.csv', index=False)
