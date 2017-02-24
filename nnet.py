import itertools
#import matplotlib.pyplot as plt
#import pydot
import math
import operator
import os
import pickle
import random
import time

import numpy as np
import theano
import theano.sandbox.cuda.nnet
import theano.tensor as T

from ai import AI
from game import Game
from verboseprint import *


class QLearningNNet(AI):

    def __init_nnet(self):

        # Build the neural network

        # Need to initialize the parameters to a small, random number
        noise = (1 / (np.sqrt(self.n_inputs * self.n_outputs *
                              self.n_hidden1 * self.n_hidden2)))

        # Weights and biases are theano shared variables, so that they can hold
        # a state
        W1 = theano.shared(noise * np.random.randn(self.n_inputs,
                                                   self.n_hidden1).astype(np.float32), name='W1')
        W2 = theano.shared(noise * np.random.randn(self.n_hidden1,
                                                   self.n_hidden2).astype(np.float32), name='W2')
        W3 = theano.shared(
            noise * np.random.randn(self.n_hidden2, self.n_outputs).astype(np.float32), name='W3')
        b1 = theano.shared(
            np.zeros(self.n_hidden1).astype(np.float32), name='b1')
        b2 = theano.shared(
            np.zeros(self.n_hidden2).astype(np.float32), name='b2')
        b3 = theano.shared(np.float32(self.n_outputs), name='b3')

        self.W = [W1, W2, W3]
        self.b = [b1, b2, b3]

        # GPU NOTE: The CPU is faster at the moment because these values are
        # not shared, so they must be transferred back and forth with every
        # call to epoch()
        x = T.matrix('x')
        y = T.vector('y')

        # forward prop
        z1 = x.dot(W1) + b1
        hidden1 = T.nnet.softplus(z1)
        z2 = hidden1.dot(W2) + b2
        hidden2 = T.nnet.softplus(z2)
        z3 = hidden2.dot(W3) + b3
        output = z3
        prediction = output
        # print(T.shape(prediction))
        rms = ((y - output)**2).sum()

        # gradients
        gW1, gb1, gW2, gb2, gW3, gb3 = T.grad(rms, [W1, b1, W2, b2, W3, b3])

        # build theano functions
        self.epoch = theano.function(inputs=[x, y],
                                     outputs=[output, rms],
                                     updates=((W1, W1 - self.alpha * gW1),
                                              (b1, b1 - self.alpha * gb1),
                                              (W2, W2 - self.alpha * gW2),
                                              (b2, b2 - self.alpha * gb2),
                                              (W3, W3 - self.alpha * gW3),
                                              (b3, b3 - self.alpha * gb3)))

        self.predict = theano.function(inputs=[x], outputs=prediction)

    def __init__(self, save_dir="data", lazy=False, exp_replay_size=200, exp_batch_size=40, goal=2048):
        """
        Initializes the class.

        Keyword args:
        - save_dir specifies the directory where things like game history and neural network snapshots will be saved.
        - lazy can be used to delay creation of the neural network to the first time that it is used. Mostly for internal use.
        """

        self.goal = goal
        self.save_dir = save_dir
        self.num_games_saved = 0
        self.in_training = True   # controls if play_move ignores epsilon

        # Counts the number of times an entry has been added into the experience replay queue,
        # including entries that overwrite previous ones
        self.exp_replay_count = 0

        self.exp_replay_size = exp_replay_size
        self.exp_batch_size = exp_batch_size

        self.exp_replay = [0 for x in range(self.exp_replay_size)]
        self.exp_replay_max = 0

        self.act = lambda x: [10 if i == x else -5 for i in range(4)]

        self.tile_stats = {}
        self.explore_moves = 0
        self.exploit_moves = 0

        # Create directories used by the class
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.isdir(self.save_dir + "/states"):
            os.mkdir(self.save_dir + "/states")

        # TODO: make some or all of these keyword arguments and save them in
        # save_state()
        (self.n_inputs, self.n_hidden1, self.n_hidden2, self.n_outputs, self.epochs, self.print_every,
         self.n_samples, self.batch, self.reg, self.alpha) = (16, 10, 10, 4, 200, 20, 200, 1, 0.01, 0.01)

        # Training flag: true if the model should be training
        self.train_mode = True
        self.update = False
        self.epsilon = 1
        self.gamma = 0.7

        self.chain = lambda mat: list(itertools.chain.from_iterable(mat))
        if not lazy:
            self.__init_nnet()
            self.initialized = True
        else:
            self.initialized = False

    def reward(self, board):
        """The basic reward function that describes the "winning" conditions. In this case, it
        incentivizes leaving squares empty, getting a new highest tile, and matching tiles."""
        a, b, c = (10, 0.2, 10)  # hyperparameter: tune at leisure
        empty_square_reward = (board.shape[0] ** 2 - np.count_nonzero(board)) ** 4
        highest_tile_reward = np.max(board.flatten())
        tile_values_reward = np.log10((board ** 2).flatten().sum())

        return a * empty_square_reward + b * highest_tile_reward + c * tile_values_reward

    def train_on_batch(self, batch):
        """Given a batch of tuples in the form (s_n, a, s_n+1, r_n+1), performs gradient descent to
        train the model. Returns None."""
        X_train = []
        y_train = []
        for old_s, a, new_s, new_r in batch:
            old_q = self.predict(old_s.reshape(1, -1))
            # print(old_q)
            new_q = self.predict(new_s.reshape(1, -1))
            maxQ = np.max(new_q)  # predicted reward for the best move after the new move
            y = old_q.copy()

            # Q-learning formula
            y[0][a] = new_r + (self.gamma * maxQ)
            X_train.append(old_s)
            y_train.append(y)

        # TODO: implement actual minibatch descent
        for x, y in zip(X_train, y_train):
            # print(x.shape, y.shape)
            self.epoch(x.reshape(1, -1), y.reshape(-1))

    def train_on_games(self, games, n_training_boards=40):
        """Picks a given amount of random positions from the given list of games, and then uses each
        of those positions to train the model. Returns None."""
        total_positions = []
        for game in games:
            boards = game.get_all_boards()
            for i, position in enumerate(boards[:-2]):  # last 2 positions are unhelpful
                next_position = boards[i+1].board.astype('float32')
                next_reward = self.reward(next_position)
                past_move = game.history[i]
                total_positions.append((position.board.astype('float32'), past_move,
                                        next_position, next_reward))
        self.train_on_batch(random.sample(total_positions, n_training_boards))
        
    def save_state(self, filename):
        if not self.initialized:
            self.__init_nnet()
            self.initialized = True

        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_state(filename):
        with open(filename, 'rb') as infile:
            obj = pickle.load(infile)
        return obj

    def __setstate__(self, state):
        if not 'W' in dir(self):
            self.__init__()

        W, b = state
        for i, w in enumerate(W):
            self.W[i].set_value(w)
        for i, b_ in enumerate(b):
            self.b[i].set_value(b_)

    def __getstate__(self):
        return ([x.get_value() for x in self.W], [x.get_value() for x in self.b])

    def play_move(self, board):
        """If self.in_training is True, uses epsilon to determine whether to make a random move or a
        calculated one. If False, always makes a calculated move."""
        if self.in_training:
            return self.training_play_move(board)
        else:
            print(board)
            print('\n')
            return self.predict_play_move(board)
    
    def predict_play_move(self, board):
        """Attempts to play the best move according to the current model."""
        return np.argmax(self.predict(board.board.reshape(1, -1).astype('float32')))

    def training_play_move(self, board):
        """Depending on epsilon, has a probability to either play a random move or play the move
        that the model thinks is best."""
        if np.random.random() < self.epsilon:
            return np.random.choice(4)
        else:
            return self.predict_play_move(board)

    def after_game_hook(self, game):

        # Update statistics fields
        max_tile = max(self.chain(game.board.board.tolist()))
        if max_tile in self.tile_stats:
            self.tile_stats[max_tile] += 1
        else:
            self.tile_stats[max_tile] = 1

        vprint("Highest tile: {}, status: {}".format(
            max_tile, game.game_status()), prefix=False)

        # save to file
        self.num_games_saved += 1
        game.save("training_games/game-{}.dat".format(self.num_games_saved))

    def print_report(self):
        vprint_np("########################")
        vprint_np("Tile statistics summary")
        vprint_np("########################")
        vprint_np("")

        total = sum([self.tile_stats[x] for x in self.tile_stats])

        for key in self.tile_stats:
            vprint_np(" * {} : {} ({}%)".format(key,
                                                self.tile_stats[key], 100 * (self.tile_stats[key] / total)))

        vprint_np("")

        vprint_np("########################")
        vprint_np("Move staticstics summary")
        vprint_np("########################")

        vprint_np("")
        vprint_np("Total moves made: {}".format(
            self.explore_moves + self.exploit_moves))
        vprint_np(" * Random moves: {}".format(self.explore_moves))
        vprint_np(" * Non-random moves: {}".format(self.exploit_moves))
