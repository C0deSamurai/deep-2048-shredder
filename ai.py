"""This class is a generalized interface that represents any algorithm or strategy for playing
2048. The default is simply printing the board to stdout and asking for help from the user!"""


from game import Game, input_player


class AI:
    """An AI that plays 2048 of some variety."""
    def __init__(self, **kwargs):
        """Kwargs get passed to a Game constructor."""
        self.kwargs = kwargs

    def play_move(self, board):
        """Given a board, makes a move. Where the magic happens."""
        # the best algorithm of all time
        return input_player(board)

    def play_game_to_completion(self, game, tiles=(2, 4), weights=(9, 1)):
        """Given a starting position in a game, makes moves until the game is completed. Returns the
        game's status at completion. After the game, implements the after_game_hook."""
        game.play_to_completion(self.play_move, tiles, weights)
        self.after_game_hook(game)
        return game.game_status()

    def after_game_hook(self, game):
        """Given a completed game, does whatever."""
        pass

    def score_trials(self, width=4, height=4, goal=2048, n_games=100, tiles=(2, 4), weights=(9, 1)):
        """Returns the percentage of games that ended in victory given a certain amount of
        trials. Tiles and weights regulate random drops."""
        num_won = 0
        for i in range(n_games):
            game = Game(width, height, goal=goal)
            if self.play_game_to_completion(game, tiles, weights) > 0:
                num_won += 1
        return num_won / n_games
