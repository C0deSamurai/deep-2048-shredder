import cProfile
import itertools
import os
import pstats
import time
from shutil import copyfile

import numpy as np

from game import Game
from nnet import QLearningNNet
from verboseprint import *


def main():
    VerbosePrint.DEBUG = True

    if not os.path.isdir("data"):
        os.mkdir("data")
        
    session_start = 11

    chain = lambda mat: list(itertools.chain.from_iterable(mat))
    for i in range(15):

        #nnet = QLearningNNet.restore_state("data/states/session{}/snapshot25".format(session_start - 1))
        nnet = QLearningNNet(goal=512)
        nnet.goal = 512
        nnet.train(100, session=session_start)
        print([x.get_value() for x in nnet.W])
        if float('nan') in chain([x.get_value().tolist() for x in nnet.W]):
            vprint("nan value detected!", msg=PRINT_FATAL)
            break
        nnet.print_report()


        session_start += 1
    
    vprint("Copying log...")
    copyfile("data/log", "nnet_log.txt")

    #nnet.train_mode = False

    #game = Game()
    #nnet.play_game_to_completion(game)

    #nnet.epoch.profile.print_summary()

PROFILE=False
if __name__=="__main__":
    if PROFILE == True:
        cProfile.run("main()", "profile.prof")

        stream = open('stats.txt', 'w');
        stats = pstats.Stats('profile.prof', stream=stream)
        stats.sort_stats('tottime')
        stats.print_stats()
        stream.close()
    else:
        main()
