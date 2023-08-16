import rules.othello
import copy
from info import info









class Node:

  def __init__(self, state, parent=None):
    self.accumulated_outcome = 0
    self.visits = 0
    self.probability = 0
    self.state = state
    self.children = []
    self.parent = parent

  def expand(self):
    outcome = self.evaluate()
    self.backpropagate(outcome)

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
      child = Node(next_state)
      self.children.append(child)

  def backpropagate(self, outcome):
    node = self
    node.visits += 1
    node.accumulated_outcome += outcome
    while node.parent:
      node.backpropagate(outcome)
      node = node.parent

  def select(self):
    child = self.children[0]
    return child

  def evaluate(self):
    return 0.7

  def simulate(self):
    node = self
    while node.children:
      node = node.select()
    node.expand()

  def _repr_helper(self, indent=0):
    # Represent current node
    lines = [' ' * indent + f'Node:']
    next_indentation = ' ' * (indent + 2)
    lines.append(next_indentation + f'accumulated_outcome: {self.accumulated_outcome}')
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

state = rules.othello.Othello()

# If the following function isn't called,
# self.state.get_legal_moves() will be an empty list.
state.initialize_board()
root = Node(state)
root.simulate()
root.simulate()
root.simulate()
root.simulate()
info(root)
