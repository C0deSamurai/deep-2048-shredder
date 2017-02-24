"""Runs a round of training where the bot plays lots and lots of games, and then reads them all in
again to learn from its mistakes. Saves the net's data to a file afterwards."""


import nnet
from game import Game

N_GAMES = 10
N_EPOCHS = 10
N_TRAINING_BOARDS = 100


agent = nnet.QLearningNNet()


for epoch in range(N_EPOCHS):
    training_games = []
    for i in range(N_GAMES):
        g = Game()
        agent.play_game_to_completion(g)  # this will just make random moves
        training_games.append(g)

    # now, we train on what we just learned
    agent.train_on_games(training_games, N_TRAINING_BOARDS)
    agent.epsilon -= (0.9 / N_EPOCHS)   # after the last one, it should be at .1


agent.save_state("trained_nnet.pkl")
print(agent.W)

g = Game()
agent.in_training = False
agent.play_game_to_completion(g)
g.print_boards()
