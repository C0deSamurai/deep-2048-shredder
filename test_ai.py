
from nnet import QLearningNNet
from game import Game
import os

import time

from verboseprint import *

import cProfile
import pstats


def main():

    if not os.path.isdir("data"):
        os.mkdir("data")

    nnet = QLearningNNet()

    nnet.train(10)

if __name__=="__main__":
    cProfile.run("main()", "profile.prof")

    stream = open('stats.txt', 'w');
    stats = pstats.Stats('profile.prof', stream=stream)
    stats.sort_stats('tottime')
    stats.print_stats()
    stream.close()