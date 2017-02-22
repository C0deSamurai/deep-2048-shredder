import numpy as np

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
        
    

    nnet = QLearningNNet(goal=256)
    
    print(nnet._value(np.array([[0, 0, 0, 256], [0, 0, 128, 128], [0, 0, 0, 0], [0, 0, 0, 0]])))
    print(nnet._value(np.array([[0, 0, 0, 256], [0, 0, 0, 256], [0, 0, 0, 0], [0, 0, 0, 0]])))
    
    nnet.train(100)
    print([x.get_value() for x in nnet.W])
    #nnet.epoch.profile.print_summary()

if __name__=="__main__":
    cProfile.run("main()", "profile.prof")

    stream = open('stats.txt', 'w');
    stats = pstats.Stats('profile.prof', stream=stream)
    stats.sort_stats('tottime')
    stats.print_stats()
    stream.close()