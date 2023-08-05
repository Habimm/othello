from info import info
import os
import pandas as pd
import plotly.graph_objects as go
import re

# Read the data from the CSV file
df = pd.read_csv('generated/training_history.csv')

# Create the plot
fig = go.Figure()

df = df.iloc[::-1]

fig.add_trace(
  go.Scatter(
      x=df['epoch'],
      y=df['loss'],
      mode='lines+markers',
      name='Mean Squared Error Loss',
      hovertemplate='Epoch: %{x}<br>MSE: %{y:.6f}'
  )
)

fig.update_layout(
    title='Mean Squared Error (MSE) Loss over Epochs',
    xaxis_title='Epoch',
    yaxis_title='Mean Squared Error Loss',
)

fig.show()




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

plays_directory = 'generated/plays/eOthello/'
play_paths = get_filepaths(plays_directory)
play_paths.sort(key=sort_key)

# Lists to store data for the new CSV
play_paths_list = []
number_of_wins_list = []
number_of_plays_list = []

# Iterate over each play path
for play_path in play_paths:
    a = pd.read_csv(play_path)
    number_of_wins = a[a['black_outcome'] == -1].shape[0]
    number_of_plays = a.shape[0]  # This gets the total number of rows in the dataframe, i.e., the total number of plays

    play_paths_list.append(play_path)
    number_of_wins_list.append(number_of_wins)
    number_of_plays_list.append(number_of_plays)

# Create a DataFrame from the lists
df = pd.DataFrame({
    'play_path': play_paths_list,
    'number_of_wins': number_of_wins_list,
    'number_of_plays': number_of_plays_list
})

# Save the DataFrame to a new CSV file
df.to_csv('generated/winning_rates.csv', index=False)

# Read the saved CSV file
df = pd.read_csv('generated/winning_rates.csv')

# Extract epoch numbers from play_path for plotting
df['epoch'] = df['play_path'].apply(lambda x: int(re.search(r'-([\d_]+)', x).group(1).replace('_', '')))

# Create the plot
fig = go.Figure()

# Calculate winning rate
winning_rate_percentage = 100 * df['number_of_wins']/df['number_of_plays']

fig.add_trace(go.Scatter(
    x=df['epoch'],
    y=winning_rate_percentage,
    mode='lines+markers',
    name='Winning Rate',
    hovertemplate='Epoch: %{x}<br>Winning Rate: %{y:.0f}%'
))

fig.update_layout(
    title='Winning Rate (over 20 plays) over Epochs',
    xaxis_title='Epoch',
    yaxis_title='Winning Rate (%)',  # Updated y-axis title to indicate percentage
)

fig.show()
