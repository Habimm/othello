from flask import Flask, request, jsonify
from flask_cors import CORS
import mcts
import rules.othello
import threading
import time
import variables_info

app = Flask(__name__)
CORS(app)

# Global variable to store the move received from the API
received_move = None
mcts_move = None

@app.route('/receive_json', methods=['POST'])
def receive_json():
    global received_move
    data = request.json
    received_move = data
    return jsonify({"message": "JSON received!"})

@app.route('/minotaurus_move', methods=['GET'])
def minotaurus_move():
    global mcts_move
    assert mcts_move is not None
    mino_move = tuple(mcts_move)
    mcts_move = None
    variables_info.d(mino_move)
    return jsonify(mino_move)

def start_flask_app():
    app.run(debug=True, port=5000, use_reloader=False)  # Disable reloader here

def play_game(model_load_path):
    global received_move   # <-- Add this
    game = rules.othello.Othello(should_draw_tiles=False)
    game.draw_board()
    game.initialize_board()
    game.current_player = 0
    other_player_has_a_move = True
    moves = []
    black_outcome = None
    initial_state_root = mcts.Node(model_load_path, game)
    initial_state_root.expand()
    current_root = initial_state_root

    while True:
        # Check if the player pointed at by `game.current_player` has a legal move.
        if game.has_legal_move():
            other_player_has_a_move = True
            move = None

            if game.current_player == 0: # == BLACK PLAYER
                # Wait until a move is received
                while received_move is None:
                    time.sleep(0.1)  # Poll every 0.1 seconds

                move = tuple(received_move)
                received_move = None  # Reset for the next move

                variables_info.d(move)
                variables_info.d(current_root.state.get_legal_moves())

                game.move = move
                game.make_move()

                current_root = current_root.next_node(move, game.current_player)

            elif game.current_player == 1: # == WHITE PLAYER

                variables_info.d(current_root)

                move = game.make_move_with_mcts(current_root)
                global mcts_move
                mcts_move = move
                current_root = current_root.next_node(move, game.current_player)

                # send `move` to othello-js

            assert move is not None
            print(move)
            moves.append(move)

        else:
            if other_player_has_a_move is False:
                black_outcome = game.get_black_outcome()
                break
            other_player_has_a_move = False

        game.current_player = 1 - game.current_player
        if not game.has_legal_move():
            game.current_player = 1 - game.current_player

        json_file_path = f'human_tree_{len(moves)}.json'
        initial_state_root.to_json(json_file_path)

if __name__ == '__main__':
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=start_flask_app)
    flask_thread.start()

    # Start the game logic
    play_game('/home/dimitri/code/othello/output/11111111111111111/0/models/batches_500.keras')
