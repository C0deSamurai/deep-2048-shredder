
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
    #print([x.get_value() for x in nnet.W])
    vprint("checking for games.txt...", msg=PRINT_SUCCESS)
    
    if os.path.isfile("data/games.txt"):
        vprint("found. removing...")
        os.remove("data/games.txt")

    time0 = time.time()

    n = 10
    staten = 0
    vprint("starting training...")
    boards = []

    avg_time = 0
    for i in range(n):
        vprint("starting game #" + str(i) + '...', end='')
        time0_ = time.time()
        game = Game()
        nnet.play_game_to_completion(game)
        game.append("data/games.txt")
        boards.append(game.board)
        time1_ = time.time()
        avg_time += time1_ - time0_
        if i % 20 == 0:
            vprint("saving state " + str(staten))
            nnet.save_state("data/states/state" + str(staten))
            staten += 1
    time1 = time.time()

    avg_time = avg_time / n
    vprint("Time taken to play " + str(n) + " games: " + str(time1-time0))
    vprint("Average time per game: " + str(avg_time))

    vprint("saving state " + str(staten))
    nnet.save_state("data/states/state" + str(staten))
    staten += 1


    games = Game.open_batch("data/games.txt")
    vprint("verifying boards...")
    boards_passed = True
    for i in range(n):
        if not games[i].board.board.tolist() == boards[i].board.tolist():
            vprint("verification failed on " + str(i), msg=PRINT_FAIL)
            boards_passed = False

    if boards_passed:
        vprint("verification success!", msg=PRINT_SUCCESS)


    #print([x.get_value() for x in nnet.W])
    vprint("Done.")

if __name__=="__main__":
    cProfile.run("main()", "profile.prof")

    stream = open('stats.txt', 'w');
    stats = pstats.Stats('profile.prof', stream=stream)
    stats.print_stats()
    stream.close()