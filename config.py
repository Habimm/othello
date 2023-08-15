from datetime import datetime
import json

# 1. Generate the timestamp
now = datetime.now()
timestamp = now.strftime("generated_%Y%m%d%H%M%S")
print(f'set -gx OTHELLO_GENERATED_PATH {timestamp}')

# 2. Configuration values from your JSON string
config = {
  "batch_size": 64,
  "epochs": 1000,
  "num_epochs_per_checkpoint": 100,
  "num_games_for_supervised_training": 10,
  "number_of_games": 20,
  "RANDOM_NUMBERS_SEQUENCE_NAME": 1
}

# 3. Print fish shell commands to set the environment variables
for key, value in config.items():
  # Convert the key to uppercase for the environment variable name
  env_var_name = f"OTHELLO_{key.upper()}"
  print(f'set -gx {env_var_name} {value}')
