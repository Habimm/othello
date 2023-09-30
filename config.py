import datetime
import os

environment_variables = {
  'batch_size': (64, int),

  # Increase this to incentivize using explorative actions
  'c_puct': (4, int),

  'epochs': (100, int),
  'num_epochs_per_checkpoint': (100, int),
  'num_games_for_supervised_training': (10, int),

  'num_processes': (1, int),
  # 'num_processes': multiprocessing.cpu_count(),

  'num_simulations': (100, int),
  'number_of_games': (1, int),
  'oracle_url': ('http://localhost:8000/predict', str),

  'output_path': ('output/20230828091759', str),
  # 'output_path': ('output/generated_20230814012642', str),
  # 'output_path': ('output/generated_20230814012642_mcts', str),
  # 'output_path': ('output/{timestamp_of_now}', str),

  'random_seed': (42, int),
}

def env(parameter_name):
  environment_variable_name = f'OTHELLO_{parameter_name.upper()}'
  value = os.environ.get(environment_variable_name)
  assert value is not None, f'Error: Environment variable {environment_variable_name} not set.'
  _, value_type = environment_variables[parameter_name]
  value = value_type(value)
  return value

if __name__ == '__main__':
  for key, (value, _) in environment_variables.items():
    env_var_name = f'OTHELLO_{key.upper()}'
    if key == 'output_path':
      now = datetime.datetime.now()
      timestamp_of_now = now.strftime('%Y%m%d%H%M%S')
      value = value.replace('{timestamp_of_now}', timestamp_of_now)
    print(f'set -gx {env_var_name} {value}')
