
from nnet import QLearningNNet
from game import Game
import os

import time

from verboseprint import *

import cProfile
import pstats


def main():
    VerbosePrint.DEBUG = True

    if not os.path.isdir("data"):
        os.mkdir("data")

    nnet = QLearningNNet()
    nnet.train(100, snapshot_every=20)

    print([x.get_value() for x in nnet.W])
    nnet.train_mode = False

    vprint("Playing 10 games to test:")
    for i in range(10):
        vprint("starting game #" + str(i) + '...', end='')
        game = Game()
        nnet.play_game_to_completion(game)

    #nnet.train(2000)

if __name__=="__main__":
    cProfile.run("main()", "profile.prof")

    stream = open('stats.txt', 'w');
    stats = pstats.Stats('profile.prof', stream=stream)
    stats.sort_stats('tottime')
    stats.print_stats()
    stream.close()