import datetime
import dotenv
import os

# Load .env file
dotenv.load_dotenv()

for key, value in os.environ.items():
  if key.startswith("OTHELLO"):
    if "{now}" in value:
      formatted_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
      value = value.replace("{now}", formatted_now)
      os.environ[key] = value

environment_variable_types = {
  'OTHELLO_BATCH_SIZE': int,
  'OTHELLO_C_PUCT': int,
  'OTHELLO_EPOCHS': int,
  'OTHELLO_NUM_EPOCHS_PER_CHECKPOINT': int,
  'OTHELLO_NUM_GAMES_FOR_SUPERVISED_TRAINING': int,
  'OTHELLO_NUM_PROCESSES': int,
  'OTHELLO_NUM_SIMULATIONS': int,
  'OTHELLO_NUMBER_OF_GAMES': int,
  'OTHELLO_ORACLE_URL': str,
  'OTHELLO_OUTPUT_PATH': str,
  'OTHELLO_RANDOM_SEED': int,
}

def parameter(environment_variable_name):
  value = os.environ.get(environment_variable_name)
  assert value is not None, f'Error: Environment variable {environment_variable_name} not set.'
  value_type = environment_variable_types[environment_variable_name]
  value = value_type(value)
  return value

