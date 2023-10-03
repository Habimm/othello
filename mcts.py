import copy
import json
import math
import multiprocessing
import numpy
import os
import requests
import rules.othello
import sys
import tensorflow
import variables_info

def get_env_variable(name):
  value = os.environ.get(name)
  if value is None:
    print(f'Error: Environment variable {name} not set.')
    sys.exit(1)
  return value

def board_to_tensor(board, player):
  # initialize the new lists with zeros
  black = [[0]*len(row) for row in board]
  white = [[0]*len(row) for row in board]

  # populate the new lists
  for row_index in range(len(board)):
    for col_index in range(len(board[row_index])):
      if board[row_index][col_index] == 1:
        black[row_index][col_index] = 1
      elif board[row_index][col_index] == 2:
        white[row_index][col_index] = 1

  assert player in [0, 1]
  if player == 0:
    neural_network_input = [black, white]
  if player == 1:
    neural_network_input = [white, black]

  board_chw = neural_network_input
  board_chw_np = numpy.array(board_chw)
  board_hwc_np = board_chw_np.transpose(1, 2, 0)
  board_hwc = board_hwc_np.tolist()

  return board_hwc

C_PUCT = int(get_env_variable('OTHELLO_C_PUCT'))
NUM_SIMULATIONS = int(get_env_variable('OTHELLO_NUM_SIMULATIONS'))
OTHELLO_ORACLE_URL = get_env_variable('OTHELLO_ORACLE_URL')
OUTPUT_PATH = get_env_variable('OTHELLO_OUTPUT_PATH')

class Node:

  def __init__(self, model_load_path, state_to_be, parent=None, move=None, model=None):
    self.state = rules.othello.Othello()
    self.state.board = copy.deepcopy(state_to_be.board)
    self.state.current_player = copy.deepcopy(state_to_be.current_player)
    self.state.num_tiles = copy.deepcopy(state_to_be.num_tiles)

    if model is None:
      self.model = tensorflow.keras.models.load_model(model_load_path, compile=False)
    else:
      self.model = model
    self.model_load_path = model_load_path
    self.accumulated_relative_outcome = 0
    self.visits = 0
    self.probability = 1
    self.children = []
    self.move = move
    self.parent = parent
    self.is_final = not self.state.has_legal_move()
    if parent is None:
      self.moving_along = True
    else:
      self.moving_along = False

    self.accumulated_relative_values = None
    self.visit_counts = None
    self.probabilities = None
    self.average_relative_values = None
    self.sum_visit_counts = None

  def expand(self):
    moves = self.state.get_legal_moves()
    for move in moves:

      # Create the board, which we want to evaluate.
      next_state = rules.othello.Othello()
      next_state.board = copy.deepcopy(self.state.board)
      next_state.current_player = copy.deepcopy(self.state.current_player)
      next_state.num_tiles = copy.deepcopy(self.state.num_tiles)
      next_state.move = move
      next_state.make_move()

      next_state.current_player = 1 - next_state.current_player
      if not next_state.has_legal_move():
        next_state.current_player = 1 - next_state.current_player
        # if this same player from the last round
        # still does not have a legal move,
        # then this next state is a final state of the game.
        # In this case, the Node() constructor will set self.is_final to True.

      child = Node(
        model_load_path=self.model_load_path,
        state_to_be=next_state,
        parent=self,
        move=move,
        model=self.model,
      )
      self.children.append(child)

    number_of_children = len(self.children)
    self.accumulated_relative_values = numpy.zeros(number_of_children, dtype=numpy.float64)
    self.average_relative_values = numpy.zeros(number_of_children, dtype=numpy.float64)
    self.visit_counts = numpy.zeros(number_of_children, dtype=numpy.int64)
    self.probabilities = numpy.ones(number_of_children, dtype=numpy.float64)
    self.sum_visit_counts = 0

  def backpropagate(self, black_outcome):
    if not self.parent: return

    self_index = self.parent.children.index(self)
    self.parent.visit_counts[self_index] += 1
    self.parent.sum_visit_counts += 1

    if self.parent.state.current_player == 0:
      self.parent.accumulated_relative_values[self_index] += black_outcome
    elif self.parent.state.current_player == 1:
      self.parent.accumulated_relative_values[self_index] -= black_outcome

    # Update the average incrementally
    # Might want to integrate this formula later: https://math.stackexchange.com/questions/106313/regular-average-calculated-accumulatively
    self.parent.average_relative_values[self_index] = self.parent.accumulated_relative_values[self_index] / self.parent.visit_counts[self_index]
    assert numpy.isclose(self.parent.average_relative_values[self_index] * self.parent.visit_counts[self_index], self.parent.accumulated_relative_values[self_index], atol=1e-8), (
      f'self.parent.average_relative_values[self_index]: {self.parent.average_relative_values[self_index]}, '
      f'self.parent.visit_counts[self_index]: {self.parent.visit_counts[self_index]}, '
      f'self.parent.accumulated_relative_values[self_index]: {self.parent.accumulated_relative_values[self_index]}'
    )

    self.parent.backpropagate(black_outcome)

  def best_move(self):
    index_of_max_visits = numpy.argmax(self.visit_counts)
    child_with_most_visits = self.children[index_of_max_visits]
    child_with_most_visits_move = child_with_most_visits.move
    return child_with_most_visits_move

  def sample_move(self):
    if self.sum_visit_counts == 0:
      return self.children[0].move
    visit_counts_distribution = self.visit_counts / self.sum_visit_counts
    chosen_child = numpy.random.choice(self.children, p=visit_counts_distribution)
    return chosen_child.move

  def next_node(self, move, current_player):
    move_child = None
    move_child_index = None
    for child_index, child in enumerate(self.children):
      if child.move == move:
        move_child = child
        move_child_index = child_index
        break

    assert move_child is not None, f"No child to make the move {move}. Expand first!"

    # This is needed, in case we do NOT use MCTS scores (using a number of simulations > 0)
    # to select a move.
    # We might, for example, select a move on behalf of the opponent.
    if self.visit_counts[move_child_index] == 0:
      assert not move_child.children, "An unvisited leaf cannot have children!"
      move_child.expand()
    move_child.parent = None
    move_child.moving_along = True
    return move_child

  def select(self):
    uct_scores = self.average_relative_values + C_PUCT * self.probabilities * numpy.sqrt(self.sum_visit_counts) / (1 + self.visit_counts)
    best_child_index = numpy.argmax(uct_scores)
    best_child = self.children[best_child_index]
    assert best_child is not None
    return best_child

  def evaluate(self):
    if self.is_final:
      black_outcome = self.state.get_black_outcome()
    else:
      move_board_tensor = board_to_tensor(self.state.board, self.state.current_player)
      prediction = self.model.predict([move_board_tensor], verbose=0)
      assert prediction.shape == (1, 1), 'Assertion failed: prediction.shape == (1, 1)'
      current_outcome = prediction[0][0]
      black_outcome = None
      if self.state.current_player == 0: black_outcome = current_outcome
      if self.state.current_player == 1: black_outcome = -current_outcome
      assert black_outcome is not None, "There are only two players: 0 (black) and 1 (white)."
    return black_outcome

  def simulate(self):
    node = self
    assert node.children, f"Every node must have children, before the first simulation is conducted!\n{node}"
    while node.children:
      node = node.select()
    black_outcome = node.evaluate()
    node.expand()
    node.backpropagate(black_outcome)

  def search(self):
    for _ in range(NUM_SIMULATIONS):
      self.simulate()

  def to_dot(self):
    lines = ['digraph MCTS {']

    def traverse(node, parent_id=None):
      node_id = f"node_{id(node)}"
      node_label = f"""
      node_id={node_id}
      accumulated_relative_values={node.accumulated_relative_values}
      average_relative_values={node.average_relative_values}
      visit_counts={node.visit_counts}
      probabilities={node.probabilities}
      sum_visit_counts={node.sum_visit_counts}
      children={[id(child) for child in node.children]}
      state={node.state}
      """

      # Conditional background coloring based on the 'moving_along' attribute
      if node.moving_along:
        if node.state.current_player == 0:
          fillcolor = 'fillcolor="#FF000080", style="filled"'
        if node.state.current_player == 1:
          fillcolor = 'fillcolor="#0000FF80", style="filled"'
      else:
        fillcolor = 'fillcolor="#FFFFFF", style="filled"'

      if node.is_final:
        fillcolor = 'fillcolor="#00FF0080", style="filled"'

      lines.append(f'{node_id} [label="{node_label}", {fillcolor}];')

      if parent_id:
        lines.append(f'{parent_id} -> {node_id};')

      for child in node.children:
        traverse(child, node_id)

    traverse(self)
    lines.append('}')
    return '\n'.join(lines)

  def _repr_helper(self, indent=0):
    # Represent current node
    lines = [' ' * indent + f'Node:']
    next_indentation = ' ' * (indent + 2)
    lines.append(next_indentation + f'accumulated_relative_values: {self.accumulated_relative_values}')
    lines.append(next_indentation + f'visit_counts: {self.visit_counts}')
    lines.append(next_indentation + f'probabilities: {self.probabilities}')
    lines.append(next_indentation + f'legal_moves: {self.state.get_legal_moves()}')
    state_representation = f'{self.state}'
    state_representation = state_representation.replace('\n', '\n' + next_indentation)
    lines.append(next_indentation + f'state: {state_representation}')

    # Represent children
    for child in self.children:
      lines.append(child._repr_helper(indent + 4))

    return '\n'.join(lines)

  def __repr__(self):
    return self._repr_helper()

  def to_json(self, file_path):
    json_array = []

    def traverse(node):
      node_dict = {
        'node_id': f"node_{id(node)}",
        'accumulated_relative_values': node.accumulated_relative_values.tolist() if node.accumulated_relative_values is not None else None,
        'average_relative_values': node.average_relative_values.tolist() if node.average_relative_values is not None else None,
        'visit_counts': node.visit_counts.tolist() if node.visit_counts is not None else None,
        'probabilities': node.probabilities.tolist() if node.probabilities is not None else None,
        'sum_visit_counts': node.sum_visit_counts,
        'children': [f"node_{id(child)}" for child in node.children],
        'state': str(node.state)
      }

      if node.moving_along:
        if node.state.current_player == 0:
          node_dict['fillcolor'] = '#FF000080'
          node_dict['style'] = 'filled'
        if node.state.current_player == 1:
          node_dict['fillcolor'] = '#0000FF80'
          node_dict['style'] = 'filled'
      else:
        node_dict['fillcolor'] = '#FFFFFF'
        node_dict['style'] = 'filled'

      if node.is_final:
        node_dict['fillcolor'] = '#00FF0080'
        node_dict['style'] = 'filled'

      json_array.append(node_dict)

      for child in node.children:
        traverse(child)

    traverse(self)

    with open(file_path, 'w') as json_file:
      json.dump(json_array, json_file, indent=2)

    return json_array  # Return the array for optional further use
