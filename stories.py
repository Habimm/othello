from info import info
import plotly.io as pio
import json
import json
import numpy
import os
import pandas as pd
import plotly.graph_objects as go
import re

generated_path = 'generated_200_plays'



# Create a tournament table.
# ===================================================================================
def get_filepaths(directory):
  filepaths = []
  for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
      filepaths.append(os.path.join(dirpath, filename))
  return filepaths

# Custom sorting function
def sort_key(filepath):
    # Extract numbers from the filename, remove underscores, and convert to an integer
    number = int(re.search(r'-([\d_]+)', filepath).group(1).replace('_', ''))
    print(number)  # Assuming info() is a print for the sake of this example
    return number

plays_directory = f'{generated_path}/plays/eOthello/'
play_paths = get_filepaths(plays_directory)
play_paths.sort(key=sort_key)

# Lists to store data for the new CSV
game_paths_list = []
number_of_wins_list = []
number_of_games_list = []

# Iterate over each play path
for play_path in play_paths:
    a = pd.read_csv(play_path)
    number_of_wins = a[a['black_outcome'] == -1].shape[0]
    number_of_games = a.shape[0]  # This gets the total number of rows in the dataframe, i.e., the total number of plays

    game_paths_list.append(play_path)
    number_of_wins_list.append(number_of_wins)
    number_of_games_list.append(number_of_games)

# Create a DataFrame from the lists
tournament_table = pd.DataFrame({
    'play_path': game_paths_list,
    'number_of_wins': number_of_wins_list,
    'number_of_games': number_of_games_list
})

# Save the DataFrame to a new CSV file
tournament_table.to_csv(f'{generated_path}/winning_rates.csv', index=False)
# ===================================================================================



# Load the existing data from the JSON file
with open('generated/parameters.json', 'r') as json_file:
  parameters = json.load(json_file)

num_epochs_per_checkpoint = parameters['num_epochs_per_checkpoint']
num_checkpoints = parameters['num_checkpoints']
checkpoint_epochs = [num_epochs_per_checkpoint for _ in range(num_checkpoints)]
checkpoint_epochs = [sum(checkpoint_epochs[:i]) for i in range(num_checkpoints)]

num_games_for_supervised_training = parameters['num_games_for_supervised_training']
num_states = parameters['num_states']
training_batch_size_per_step = parameters['training_batch_size_per_step']

# Read the saved CSV file
tournament_table = pd.read_csv(f'{generated_path}/winning_rates.csv')

# Extract epoch numbers from play_path for plotting
tournament_table['epoch'] = tournament_table['play_path'].apply(lambda x: int(re.search(r'-([\d_]+)', x).group(1).replace('_', '')))

# Calculate winning rate
tournament_table['percentage'] = tournament_table['number_of_wins'] / tournament_table['number_of_games']

# Read the data from the CSV file
training_table = pd.read_csv(f'{generated_path}/training_history.csv')

# Create the plot
fig = go.Figure()

training_table = training_table.iloc[::-1]

# Add the scatter plot for Mean Squared Error loss
fig.add_trace(
  go.Scatter(
    x=training_table['epoch'],
    y=training_table['loss'],
    mode='lines+markers',
    name='MSE loss',
    hovertemplate='Epoch: %{x}<br>MSE: %{y:.6f}'
  )
)

# Extract winning rates for the given epochs
bar_y_values = [tournament_table.loc[tournament_table['epoch'] == num_epochs, 'percentage'].iloc[0] for num_epochs in checkpoint_epochs]
bigger = numpy.array(bar_y_values)*100

# Add the bar plot for Winning Rate
fig.add_trace(
  go.Bar(
    x=checkpoint_epochs,
    y=bar_y_values,
    customdata=bigger,
    yaxis='y2',  # Reference the secondary y-axis
    name='Winning rate',
    marker_color='rgba(220,20,60,0.5)',  # Crimson color with 50% transparency
    width=40,  # Width of the bars, adjust as necessary
    hovertemplate='Epoch: %{x}<br>Winning rate: %{customdata:.0f}%'
  )
)

num_tournament_games = None
if (tournament_table['number_of_games'] == tournament_table['number_of_games'][0]).all():
  num_tournament_games = tournament_table['number_of_games'][0]

fig.update_layout(
  legend=dict(
    x=1,
    y=1,
    xanchor='right',
    yanchor='top',
    bgcolor='rgba(255, 255, 255, 0.5)',  # Semi-transparent white background
    bordercolor='black',
    borderwidth=1
  ),
  title=f'<b>Mean squared error (MSE) loss</b> of training on {num_games_for_supervised_training} games ({num_states:,} states).<br><b>Winning rate</b> based on {num_tournament_games} games.',
  xaxis_title=f'Epoch (with a batch size of {training_batch_size_per_step})',
  yaxis_title='MSE loss',
  yaxis2=dict(  # Secondary y-axis
    title='Winning rate against random baseline',
    titlefont=dict(color="blue"),
    tickfont=dict(color="blue"),
    overlaying='y',
    side='right',
    range=[0, 1],
    tickformat='.0%',
  )
)

# fig.show()
fig.write_image("training_and_tournament.png", scale=4)
# pio.write_image(fig, 'training_and_tournament.svg')
pio.write_html(fig, file='training_and_tournament.html', auto_open=True)
