import _othello_environment
import ast
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

def train(prediction_prompts_path, models_path):
  experiment = comet_ml.Experiment(
      api_key="naHPlykXz5yy1LscJvnkLiyBJ",
      project_name='eOthello-1',
  )

  # Metrics from this training run will now be
  # available in the Comet UI

  batch_size = _othello_environment.parameter('OTHELLO_BATCH_SIZE')
  epochs = _othello_environment.parameter('OTHELLO_EPOCHS')
  num_steps_per_checkpoint = _othello_environment.parameter('OTHELLO_NUM_STEPS_PER_CHECKPOINT')
  random_seed = _othello_environment.parameter('OTHELLO_RANDOM_SEED')

  # Load the decomposed games data.
  othello_actions_dataframe = pandas.read_csv(
    prediction_prompts_path,
    sep=';',
    index_col=['Game', 'Step'],
  )

  # Fix the sequence of random numbers.
  numpy.random.seed(random_seed)
  random.seed(random_seed)
  tensorflow.random.set_seed(random_seed)

  # Define the oracle neural network's architecture
  inputs = tensorflow.keras.Input(shape=(8, 8, 2))
  x = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', data_format='channels_last')(inputs)
  x = tensorflow.keras.layers.Flatten()(x)
  outputs = tensorflow.keras.layers.Dense(1, activation='tanh')(x)
  model = tensorflow.keras.Model(inputs=inputs, outputs=outputs)
  model.summary()
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tensorflow.keras.metrics.MeanAbsoluteError()])
  filepath = f'{models_path}/batches_{{batches}}.keras'

  model_json_string = model.to_json()
  model_config_dict = json.loads(model_json_string)

  # Save training parameter in a JSON file.
  # Try to load the existing data from the JSON file
  try:
    with open(f'{models_path}/meta.json', 'r') as json_file:
      data = json.load(json_file)
  except FileNotFoundError:
    # If the file doesn't exist, initialize data as an empty dictionary
    data = {}

  # Add the new key-value pair
  data['plays_path'] = prediction_prompts_path
  data['training_epochs'] = epochs
  data['training_batch_size_per_step'] = batch_size
  data['num_steps_per_checkpoint'] = num_steps_per_checkpoint
  # + 1 because we also have the checkpoint at 0 epochs,
  # before training starts
  data['num_checkpoints'] = epochs // num_steps_per_checkpoint + 1
  data['model_config_dict'] = model_config_dict

  # Save the updated data back to the JSON file
  with open(f'{models_path}/meta.json', 'w') as json_file:
    json.dump(data, json_file, indent=2)

  # Use ast.literal_eval to convert strings to lists, then convert lists to numpy arrays
  boards = othello_actions_dataframe['Board']
  boards = boards.apply(ast.literal_eval)
  boards = boards.apply(numpy.array)
  boards = boards.tolist()
  boards = numpy.array(boards)
  boards_nhwc = tensorflow.transpose(boards, [0, 2, 3, 1])

  class SaveModelsCallback(tensorflow.keras.callbacks.Callback):
    def __init__(self):
        super(SaveModelsCallback, self).__init__()
        self.last_filepath = None
        self.global_step = 0  # Keep track of total number of steps

    def on_batch_begin(self, batch, logs=None):
        # Save every `num_steps_per_checkpoint` steps
        if self.global_step % num_steps_per_checkpoint == 0:
            readable_step = f'{self.global_step:_}'
            self.last_filepath = filepath.format(batches=readable_step)
            directory_path = os.path.dirname(self.last_filepath)
            os.makedirs(directory_path, exist_ok=True)
            self.model.save(self.last_filepath)

        self.global_step += 1  # Increment the global step count after saving

    def on_epoch_end(self, epoch, logs=None):
        # If you still want to save after the last epoch, you can leave this method as is
        if epoch == epochs - 1:
            readable_step = f'{self.global_step:_}'
            self.last_filepath = filepath.format(batches=readable_step)
            directory_path = os.path.dirname(self.last_filepath)
            os.makedirs(directory_path, exist_ok=True)
            self.model.save(self.last_filepath)

  # Prepare callback to save model every `num_steps_per_checkpoint` epochs
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
      existing_df = pandas.read_csv(f'{models_path}/training_history.csv')
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
      'mean_absolute_error': [final_accuracy],
  })

  history_df = pandas.concat([history_df, final_metrics_df])

  # 2. Concatenate the new data with the existing data
  combined_df = pandas.concat([existing_df, history_df], ignore_index=True)

  # Sort the data in reverse order (optional)
  combined_df = combined_df.iloc[::-1]

  # 3. Save the combined dataframe back to the CSV file
  combined_df.to_csv(f'{models_path}/training_history.csv', index=False)

  return checkpoint_callback.last_filepath

if __name__ == '__main__':
  train(
    prediction_prompts_path = f'{output_path}/prediction_prompts/100.keras.csv',
    model_name = 'eOthello-1',
  )
