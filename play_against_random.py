'''
    Siyan
    CS5001
    Fall 2018
    November 30, 2018
'''

from info import info
import rules.othello

if __name__ == '__main__':
    game = rules.othello.Othello(should_draw_tiles=False)
    game.draw_board()
    game.initialize_board()

    # Starts playing the game
    # The user makes a move by clicking one of the squares on the board
    # The computer makes a random legal move every time
    # Game is over when there are no more lagal moves or the board is full

    # game.run()

    # 0 = BLACK PLAYER
    # 1 = WHITE PLAYER
    game.current_player = 0
    other_player_has_a_move = True
    moves = []
    while True:
        # Check if the player pointed at by `game.current_player` has a legal move.
        # So, if there is a legal move from the current game state (board + player).
        if game.has_legal_move():
            other_player_has_a_move = True
            move = None
            if game.current_player == 0:
                # BLACK is random
                move = game.make_random_move()
            if game.current_player == 1:
                # WHITE is trained
                move = game.make_move_with_best_value()
            assert move is not None
            moves.append(move)
        else:
            if other_player_has_a_move is False:
                game.report_result()
                break
            other_player_has_a_move = False

        game.current_player = 1 - game.current_player

    info(moves)
