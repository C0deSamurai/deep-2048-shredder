import numpy as np
import theano
import theano.tensor as T
#import matplotlib.pyplot as plt
#import pydot
import math

import itertools
import operator
import time
import random

import ai
import board



class QLearningNNet(ai.AI):

    def __init__(self):
        # Set parameters
        (self.n_inputs, self.n_hidden1, self.n_hidden2, self.n_outputs, self.epochs, self.print_every, 
        self.n_samples, self.batch, self.reg, self.alpha) = (20, 8, 5, 1, 200, 20, 200, 1, 0.01, 0.5)
        
        # Training flag: true if the model should be training
        self.train = True
        self.update = False
        self.epsilon = 1
        self.gamma = 0.1
        
        ## Build the neural network
        
        # Need to initialize the parameters to a small, random number
        noise = (1/(np.sqrt(self.n_inputs * self.n_outputs * self.n_hidden1 * self.n_hidden2))) * np.random.random()

        # Weights and biases are theano shared variables, so that they can hold a state
        W1 = theano.shared(noise * np.random.randn(self.n_inputs, self.n_hidden1), name='W1')
        W2 = theano.shared(noise * np.random.randn(self.n_hidden1, self.n_hidden2), name='W2')
        W3 = theano.shared(noise * np.random.randn(self.n_hidden2), name='W3')
        b1 = theano.shared(np.zeros(self.n_hidden1), name='b1')
        b2 = theano.shared(np.zeros(self.n_hidden2), name='b2')
        b3 = theano.shared(np.float64(self.n_outputs), name='b3')

        x = T.dmatrix('x')
        y = T.dvector('y')

        # forward prop
        z1 = x.dot(W1) + b1
        hidden1 = T.nnet.softplus(z1)
        z2 = hidden1.dot(W2) + b2
        hidden2 = T.nnet.softplus(z2)
        z3 = hidden2.dot(W3) + b3
        output = z3
        prediction = output

        rms = ((y - output)**2).sum().sqrt()

        # gradients
        gW1, gb1, gW2, gb2, gW3, gb3 = T.grad(rms, [W1, b1, W2, b2, W3, b3])
        print(gb3.type())
        # build theano functions
        self.epoch = theano.function(inputs = [x, y],
                                outputs = [output, rms],
                                updates = ((W1, W1 - self.alpha * gW1),
                                           (b1, b1 - self.alpha * gb1),
                                           (W2, W2 - self.alpha * gW2),
                                           (b2, b2 - self.alpha * gb2),
                                           (W3, W3 - self.alpha * gW3),
                                           (b3, b3 - self.alpha * gb3)))

        self.predict = theano.function(inputs=[x], outputs=prediction)
        #up, down, left, right
        pass
        
    
    def train(self):
        # play a number of games
        pass
    
    def play_move(self, board):
        # TODO: change the value function to something better
        # state is what is given as input to the network
        a = 0.3
        value = lambda state: math.sqrt(sum([x**2 for x in state]) / len(state)) - (a * (sum([np.sign(x) for x in state])/2)**2)
        chain = lambda mat: list(itertools.chain.from_iterable(mat))

        if self.train and self.update:
            Sprime = chain(board.board.tolist())
            reward = value(Sprime) - value(self.current_state)
            Qprime = (self.predict([Sprime + [1, 0, 0, 0]]), self.predict([Sprime + [0, 1, 0, 0]]),\
                      self.predict([Sprime + [0, 0, 1, 0]]), self.predict([Sprime + [0, 0, 0, 1]]))
            if board.game_status() == -1:
                reward = -1000
                status = 0
            elif board.game_status(goal=256) == 1:
                reward = 1000
                status = 0
            maxQ = np.max(Qprime)
            update = reward + (self.gamma * maxQ)
            
            out, rms = self.epoch(np.array([self.current_state + [1 if i == self.last_action else 0 for i in range(4)]]), np.array([update]))
            print('updating nnet...', rms)
            self.update = False
    
        # For every move, adjust the weights
        S = chain(board.board.tolist())
        qval = (self.predict([S + [1, 0, 0, 0]]), self.predict([S + [0, 1, 0, 0]]),\
                self.predict([S + [0, 0, 1, 0]]), self.predict([S + [0, 0, 0, 1]]))
                
        
        if self.train and random.random() < self.epsilon:
            action = np.random.randint(0,4)
        else:
            action = np.argmax(qval)

        self.update = True
        self.current_state = S
        self.last_action = action
        
        #board.make_move(action)
        return action
                
        
    def after_game_hook(self, game):
        print("FINISHED")
        print(game.board)