import rules.othello
import math
import copy
from info import info






NUM_SIMULATIONS = 20
c_puct = 0.35

class Node:

  def __init__(self, state, parent=None):
    self.accumulated_outcome = 0
    self.visits = 0
    self.probability = 1
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
      child = Node(next_state, self)
      self.children.append(child)

  def backpropagate(self, outcome):
    self.visits += 1
    self.accumulated_outcome += outcome
    if self.parent:
      self.parent.backpropagate(outcome)

  def select(self):
    visits_sum = sum(child.visits for child in self.children)
    visits_sum = math.sqrt(visits_sum)
    best_score = float('-inf')
    best_child = None
    for child in self.children:
      score = c_puct * child.probability * visits_sum / (1 + child.visits)
      if score > best_score:
        best_score = score
        best_child = child

    assert best_child is not None

    return best_child

  def evaluate(self):
    return 0.7

  def simulate(self):
    node = self
    while node.children:
      node = node.select()
    node.expand()

  def search(self):
    for _ in range(NUM_SIMULATIONS):
      self.simulate()

  def to_dot(self):
    lines = ['digraph Tree {']

    def traverse(node, parent_id=None):
      node_id = f"node_{id(node)}"
      node_label = f"accumulated_outcome={node.accumulated_outcome}\nvisits={node.visits}\nprobability={node.probability}\nstate={node.state}"
      lines.append(f'{node_id} [label="{node_label}"];')

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
root = Node()
root.search()

# Write the dot string to a file
with open('tree.dot', 'w') as f:
    f.write(root.to_dot())
