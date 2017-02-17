"""This extends the AI class to make an AI that uses a generalized heuristic function with normal
tree-style search."""


import numpy as np

from ai import AI


class TreeSearchAI(AI):
    """An AI that tries to maximize a heuristic cost function that looks into the future. It also
    intelligently saves games so you can see it play."""
    def __init__(self, cost_func, ply=3, save_games=False):
        """Cost_func takes in a numpy matrix and returns a value such that higher values indicate a worse
        position. If save_games is True, saves every game. Ply is the number of moves to look
        forward. It must be at least 1.

        Memoizing the cost function could be helpful.
        """
        self.cost_func = cost_func
        self.save_games = save_games
        self.ply = ply
        self.games_played = 0

    def __helper_play(self, board, ply):
        """Looks the given amount ahead and returns the tuple (move, cost) indicating the best
        choice. Used as a recursive backbone for play()."""
        if ply == 0:  # just return this cost
            return self.cost_func(board)
        else:  # recurse and find optimal play
            new_game_states = []
            for move in (0, 1, 2, 3):
                new_game_state, is_doing_something = board.show_move(move)
                if not is_doing_something:
                    continue  # don't let the AI play moves that do nothing
                # now look ahead 1 fewer moves than before, get the optimal cost, and use that
                new_game_states.append((move, self.__helper_play(new_game_state, ply-1)))
            return min(new_game_states, key=lambda x: x[1])

    def play_move(self, board):
        """Searches into the future of all possible game states and finds the outcome that best
        minimizes the cost function, assuming that the spawner of tiles is adversarial: that is,
        that the tile spawns are designed to be as difficult as possible. Ply is the number of moves
        to look ahead. Returns a number 0-3 indicating the best move."""
        return self.__helper_play(board, self.ply)[0]

    def after_game_hook(self, game):
        if self.save_games:
            game.save("game-{}.dat".format(self.games_played))
            self.games_played += 1
        print("Working...")


# example cost function: this is really bad, though!
def neg_square_cost_function(board):
    """Returns the negative of the sum of the squares of the value of each tile, a way to encourage
    matching. Matches lost boards as 10^10 cost so you avoid them, and won boards as -10^10 so you
    get to them!"""
    if board.game_status() == -1:
        return 10 ** 10
    elif board.game_status() == 1:
        return -(10 ** 10)
    return -np.sum(board.board.flatten() ** 2)


if __name__ = "__main__":
    tree_ai = TreeSearchAI(neg_square_cost_function, ply=4, save_games=True)
    print(tree_ai.score_trials(n_games=10, goal=2048))
