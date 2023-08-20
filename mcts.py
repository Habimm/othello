from info import info
import copy
import math
import numpy
import os
import rules.othello
import tensorflow.keras.models

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

  if player == 0:
    neural_network_input = [black, white]
  if player == 1:
    neural_network_input = [white, black]
  return neural_network_input

OUTPUT_PATH = get_env_variable('OTHELLO_OUTPUT_PATH')
NUM_SIMULATIONS = int(get_env_variable('OTHELLO_NUM_SIMULATIONS'))
C_PUCT = int(get_env_variable('OTHELLO_C_PUCT'))

model_path = f'{OUTPUT_PATH}/models/eOthello-1/1_000.keras'
class Node:

  def __init__(self, state_to_be, parent=None, move=None):
    self.state = rules.othello.Othello()
    self.state.board = copy.deepcopy(state_to_be.board)
    self.state.current_player = copy.deepcopy(state_to_be.current_player)
    self.state.num_tiles = copy.deepcopy(state_to_be.num_tiles)

    self.accumulated_relative_outcome = 0
    self.visits = 0
    self.probability = 1
    self.children = []
    self.move = move
    self.parent = parent
    self.is_final = not self.state.has_legal_move()
    self.opponent_model = tensorflow.keras.models.load_model(model_path)
    if parent is None:
      self.moving_along = True
    else:
      self.moving_along = False

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

      child = Node(next_state, self, move)
      self.children.append(child)

  def backpropagate(self, black_outcome):
    self.visits += 1
    assert self.is_final or self.children, f"No children, although we have visited this node! {self}"
    assert self.state.current_player in [0, 1], "When accumulating relative outcome, the player is either 0 (black) or 1 (white)."
    if self.state.current_player == 0: self.accumulated_relative_outcome += black_outcome
    if self.state.current_player == 1: self.accumulated_relative_outcome -= black_outcome
    if self.parent:
      self.parent.backpropagate(black_outcome)

  def best_move(self):
    visits = [child.visits for child in self.children]
    index_of_max_visits = numpy.argmax(visits)
    child_with_most_visits = self.children[index_of_max_visits]
    child_with_most_visits_move = child_with_most_visits.move
    return child_with_most_visits_move

  def next_node(self, move, current_player):
    self.state.current_player = current_player
    move_child = None
    for child in self.children:
      if child.move == move:
        move_child = child
        break

    assert move_child is not None, f"No child to make the move {move}. Expand first!"

    # This is needed, in case we do NOT use MCTS scores (using a number of simulations > 0)
    # to select a move.
    # We might, for example, select a move on behalf of the opponent.
    if move_child.visits == 0:
      assert not move_child.children, "An unvisited leaf cannot have children!"
      move_child.expand()
    move_child.parent = None
    move_child.moving_along = True
    return move_child

  def select(self):
    squared_sum_of_visits = sum(child.visits for child in self.children)
    squared_sum_of_visits = math.sqrt(squared_sum_of_visits)
    best_score = float('-inf')
    best_child = None
    for child in self.children:
      score = child.accumulated_relative_outcome / (1 + child.visits) + C_PUCT * child.probability * squared_sum_of_visits / (1 + child.visits)
      if score > best_score:
        best_score = score
        best_child = child

    assert best_child is not None
    return best_child

  def evaluate(self):
    if self.is_final:
      black_outcome = self.state.get_black_outcome()
    else:
      move_board_tensor = board_to_tensor(self.state.board, self.state.current_player)
      evaluation = self.opponent_model.predict([move_board_tensor], verbose=0)
      current_outcome = evaluation[0][0]
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
    lines = ['digraph Tree {']

    def traverse(node, parent_id=None):
      node_id = f"node_{id(node)}"
      node_label = f"average_outcome={node.accumulated_relative_outcome / (node.visits + 1)}\naccumulated_relative_outcome={node.accumulated_relative_outcome}\nvisits={node.visits}\nprobability={node.probability}\nstate={node.state}"

      # Conditional background coloring based on the 'moving_along' attribute
      if node.moving_along:
        if node.state.current_player == 0:
          fillcolor = 'fillcolor="#FF000080", style="filled"'
        if node.state.current_player == 1:
          fillcolor = 'fillcolor="#0000FF80", style="filled"'
      else:
        fillcolor = ''

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
    lines.append(next_indentation + f'accumulated_relative_outcome: {self.accumulated_relative_outcome}')
    lines.append(next_indentation + f'visits: {self.visits}')
    lines.append(next_indentation + f'probability: {self.probability}')
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
