
from nnet import QLearningNNet
from game import Game

if __name__ == "__main__":
    nnet = QLearningNNet()
    for i in range(10):
        game = Game()
        nnet.play_game_to_completion(game)
        game.append("games.txt")