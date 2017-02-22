import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda.nnet
#import matplotlib.pyplot as plt
#import pydot
import math

import itertools
import operator
import time
import random


import board
from ai import AI
from game import Game

import os
import pickle

from verboseprint import *

class QLearningNNet (AI):


    def __init_nnet(self):
        
        ## Build the neural network
        
        # Need to initialize the parameters to a small, random number
        noise = (1/(np.sqrt(self.n_inputs * self.n_outputs * self.n_hidden1 * self.n_hidden2)))

        # Weights and biases are theano shared variables, so that they can hold a state
        W1 = theano.shared(noise * np.random.randn(self.n_inputs, self.n_hidden1).astype(np.float32), name='W1')
        W2 = theano.shared(noise * np.random.randn(self.n_hidden1, self.n_hidden2).astype(np.float32), name='W2')
        W3 = theano.shared(noise * np.random.randn(self.n_hidden2).astype(np.float32), name='W3')
        b1 = theano.shared(np.zeros(self.n_hidden1).astype(np.float32), name='b1')
        b2 = theano.shared(np.zeros(self.n_hidden2).astype(np.float32), name='b2')
        b3 = theano.shared(np.float32(self.n_outputs), name='b3')

        self.W = [W1, W2, W3]
        self.b = [b1, b2, b3]


        ### GPU NOTE: The CPU is faster at the moment because these values are not shared, so they must be transferred back and forth with every call to epoch()
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

        rms = ((y - output)**2).sum()

        # gradients
        gW1, gb1, gW2, gb2, gW3, gb3 = T.grad(rms, [W1, b1, W2, b2, W3, b3])

        # build theano functions
        self.epoch = theano.function(inputs = [x, y],
                                outputs = [output, rms],
                                updates = ((W1, W1 - self.alpha * gW1),
                                           (b1, b1 - self.alpha * gb1),
                                           (W2, W2 - self.alpha * gW2),
                                           (b2, b2 - self.alpha * gb2),
                                           (W3, W3 - self.alpha * gW3),
                                           (b3, b3 - self.alpha * gb3)),
                                profile=True)

        self.predict = theano.function(inputs=[x], outputs=prediction)

    def __init__(self, save_dir="data", lazy=False, exp_replay_size=200, exp_batch_size=10, goal=2048):
        """
        Initializes the class.

        Keyword args:
        - save_dir specifies the directory where things like game history and neural network snapshots will be saved.
        - lazy can be used to delay creation of the neural network to the first time that it is used. Mostly for internal use.
        """
        
        self.goal = goal
        self.save_dir = save_dir
        
        # Counts the number of times an entry has been added into the experience replay queue, 
        # including entries that overwrite previous ones
        self.exp_replay_count = 0
        
        self.exp_replay_size = exp_replay_size
        self.exp_batch_size = exp_batch_size

        self.exp_replay = [0 for x in range(self.exp_replay_size)]
        self.exp_replay_max = 0
        
        # Create directories used by the class
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.isdir(self.save_dir + "/states"):
            os.mkdir(self.save_dir + "/states")

        ### TODO: make some or all of these keyword arguments and save them in save_state()
        (self.n_inputs, self.n_hidden1, self.n_hidden2, self.n_outputs, self.epochs, self.print_every, 
        self.n_samples, self.batch, self.reg, self.alpha) = (20, 8, 5, 1, 200, 20, 200, 1, 0.01, 0.01)
        
        # Training flag: true if the model should be training
        self.train_mode = True
        self.update = False
        self.epsilon = 1
        self.gamma = 0.7

        self.chain = lambda mat: list(itertools.chain.from_iterable(mat))
        if lazy == False:
            self.__init_nnet()
            self.initialized = True
        else:
            self.initialized = False
        
    
    def per_move_callback(self):
        if self.exp_replay_count > self.exp_replay_size:
                
            #vprint("Training minibatch...")
            exp_replay_copy = []
            exp_replay_copy[:] = self.exp_replay[:]
            exp_batch = []
            while len(exp_batch) < self.exp_batch_size:
                exp_batch.append(exp_replay_copy.pop(np.random.randint(0, len(exp_replay_copy))))
            
            x_train = []
            y_train = []
            for state, action, new_state, reward in exp_batch:
                Qprime = (self.predict([new_state + [1, 0, 0, 0]]), self.predict([new_state + [0, 1, 0, 0]]),\
                          self.predict([new_state + [0, 0, 1, 0]]), self.predict([new_state + [0, 0, 0, 1]]))
                
                maxQ = np.max(Qprime)
                
                update = reward + (self.gamma * maxQ)
                x_train.append(state + [1 if i == action else 0 for i in range(4)])
                y_train.append(update)
                
            # train with mini_batch
            out, rms = self.epoch(np.array(x_train).astype(np.float32), 
                              np.array(y_train).astype(np.float32))
                              
            #vprint("finished training minibatch")
    
    def _value(self, state):
        """State should be a numpy array"""
        reciprocal = lambda x: 100/(x+0.01)
        
        _state = state
        
        # sum squares of differences of rows, 1-0, 2-1, 3-2
        _sum = ((_state[1] - _state[0])**2).sum() + ((_state[2] - _state[1])**2).sum() + ((_state[3] - _state[2])**2).sum()
        _state = _state.T
        _sum += ((_state[1] - _state[0])**2).sum() + ((_state[2] - _state[1])**2).sum() + ((_state[3] - _state[2])**2).sum()
        _state = _state.T
        #return reciprocal(sum)
        
        
        # TODO: change the value function to something better
        # state is what is given as input to the network
        a = 0.1
        tile_rms = lambda s: math.sqrt(sum([x**2 for x in s]) / len(s)) #- (a * (sum([np.sign(x) for x in _state])/2)**2)
        return reciprocal(_sum) + (a * tile_rms(self.chain(_state.tolist())))
        
    def train(self, n, snapshot_every=20, printouts=True, session=0):
        """
        Plays through n games, saving a snapshot of the neural network every `save_every` games.
        save_every can be set to None if you do not want to save snapshots of the weights.

        The neural network can be restored from this save file using restore_state().

        Has verbose printouts which can be disabled via the printouts parameter
        """
        
        # Perform lazy initialization if needed
        if self.initialized == False:
            self.__init_nnet()
            self.initialized = True

        # Reinitialize experience replay data
        self.exp_replay_count = 0
        self.exp_replay = [0 for x in range(self.exp_replay_size)]
        
        # Make sure the data save location is set up approprately
        if not os.path.isdir("data"):
            vprint("creating folder ./data...")
            os.mkdir("data")
            
        vprint("checking for games.txt...", msg=PRINT_SUCCESS)
        if os.path.isfile("data/games.txt"):
            vprint("found. removing...")
            os.remove("data/games.txt")

            
            
        time0 = time.time()

        staten = 0
        vprint("starting training...")
        boards = []

        for i in range(n):
            vprint("starting game #" + str(i) + '...', end='')
            game = Game()
            game.goal = self.goal
            self.play_game_to_completion(game)
                
            game.append("data/games.txt")
            boards.append(game.board)

            # save neural network state
            if i % snapshot_every == 0:
                vprint("saving state " + str(staten))
                self.save_state("data/states/snapshot" + str(staten))
                staten += 1
            if self.epsilon > 0.1:
                self.epsilon -= 1/n

        time1 = time.time()

        vprint("Time taken to play " + str(n) + " games: " + str(time1-time0))
        vprint("Average time per game: " + str((time1-time0) / n))

        vprint("saving final state " + str(staten))
        self.save_state("data/states/snapshot" + str(staten))
        staten += 1

        # Verify integrity of the save file games.txt
        # It could probably be done faster with hashes, but this way ensures 
        # that the class itself is functioning correctly.
        games = Game.open_batch("data/games.txt")
        vprint("verifying boards...")
        boards_passed = True
        for i in range(n):
            if not games[i].board.board.tolist() == boards[i].board.tolist():
                vprint("verification failed on " + str(i), msg=PRINT_FAIL)
                boards_passed = False

        if boards_passed:
            vprint("verification success!", msg=PRINT_SUCCESS)


        vprint("Done.")


    def save_state(self, filename):
        if self.initialized == False:
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
        if self.initialized == False:
            self.__init_nnet()
            self.initialized = True
            
        value = self._value

        if self.train_mode and self.update:
                
        
            Sprime = self.chain(board.board.tolist())
            reward = value(board.board) - value(self.current_state_np)
            Qprime = (self.predict([Sprime + [1, 0, 0, 0]]), self.predict([Sprime + [0, 1, 0, 0]]),\
                      self.predict([Sprime + [0, 0, 1, 0]]), self.predict([Sprime + [0, 0, 0, 1]]))
            if board.game_status(goal=self.goal) == -1:
                reward = -100 * (self.goal - np.max(self.chain(board.board.tolist())))
                status = 0
            elif board.game_status(goal=self.goal) == 1:
                reward = 10000
                status = 0
            maxQ = np.max(Qprime)
            update = reward + (self.gamma * maxQ)
            
            out, rms = self.epoch(np.array([self.current_state + [1 if i == self.last_action else 0 for i in range(4)]]).astype(np.float32), 
                                  np.array([update]).astype(np.float32))
                                  
            self.exp_replay[self.exp_replay_count % self.exp_replay_size] = (self.current_state, self.last_action, Sprime, reward)
            self.exp_replay_count += 1
                
                
            #vprint('updating nnet...' + str(rms))
            self.update = False
    
        # For every move, adjust the weights
        S = self.chain(board.board.tolist())
        qval = (self.predict([S + [1, 0, 0, 0]]), self.predict([S + [0, 1, 0, 0]]),\
                self.predict([S + [0, 0, 1, 0]]), self.predict([S + [0, 0, 0, 1]]))
                
                
        
        if self.train_mode and random.random() < self.epsilon:
            action = np.random.randint(0,4)
        else:
            action = np.argmax(qval)

        self.update = True
        self.current_state = S
        self.current_state_np = board.board
        self.last_action = action
        
        #board.make_move(action)
        return action
                
        
    def after_game_hook(self, game):
        vprint("Highest tile: {}, status: {}".format(max(self.chain(game.board.board.tolist())), game.game_status()), prefix=False)
        
        pass