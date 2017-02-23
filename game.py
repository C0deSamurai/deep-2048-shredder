"""This file has a class that allows someone to play a complete 2048 game with full game history."""

import os


from random import choice

import numpy as np

from board import Board

from verboseprint import *

class Game:
    """A game of 2048."""
    def __init__(self, width=4, height=4, board=None, board_obj=None, goal=2048):
        """Initializes a game with the given board. If board_obj is None, initializes a board with the given
        width, height, and optional numpy array for the actual squares. If board_obj is given, uses
        that instead: use this for subclasses of Board.

        Goal is the desired end tile to use for checking. If this is None, will play until a loss.

        """
        if board_obj is not None:
            self.board = board
        else:
            self.board = Board(width, height, board)

        self.original_board = np.copy(self.board.board)  # to make history canonical
        self.history = []
        self.spawns = []

        self.goal = goal

    def __str__(self):
        return str(self.board)

    def update_history(self, spawns, move_list):
        """Given a list of spawns and moves, takes turns spawning a tile at the given location, and
        making the desired move."""
        for i in range(len(spawns)):
            self.spawns.append(spawns[i])
            self.board.set_tile(*spawns[i][0], spawns[i][1])
            if i < len(move_list):
                self.make_move(move_list[i])

    def make_move(self, direction):
        """Direction is 0-3 clockwise from the top (0 = UP, 1 = RIGHT, etc.). Makes a move, saves it to the
        history, and updates the board. Stalling moves return 0 and do not update the history; moves
        that change something return 1.
        """
        if self.board.make_move(direction):
            self.history.append(direction)
            return 1
        else:
            return 0

    def game_status(self):
        """Returns 0 if game is still going on, 1 for victory, and -1 for loss."""
        return self.board.game_status(self.goal)

    def add_random_tile(self, tiles=(2, 4), weights=(9, 1)):
        """Adds a new tile at a randomly chosen empty spot according to the given tiles and
        corresponding weights. The default is the original 2048's default: 90% chance of a 2, and
        10% chance of a four. Logs in history so the game can be replayed.

        Does not wrap the Board's class, because this needs to be stored. Instead, manually sets a
        new tile."""

        unoccupied_slots = []
        for x in range(self.board.width):
            for y in range(self.board.height):
                if self.board.get_tile(x, y) == 0:
                    unoccupied_slots.append((x, y))

        if len(unoccupied_slots) == 0:
            vprint("Tried to add tile to full board", debug=True)
            return

        chosen_slot = choice(unoccupied_slots)
        
        normed_weights = [weight / sum(weights) for weight in weights]
        chosen_tile = np.random.choice(tiles, p=normed_weights)

        self.board.set_tile(*chosen_slot, chosen_tile)
        self.spawns.append((chosen_slot, chosen_tile))

    def play(self, move_generation_func, tiles=(2, 4), weights=(9, 1)):
        """Make a single move using the given move generation function, and then adds a new random
        tile. Returns the current game status after the move and placement. The function takes in a
        Board and returns 0-3. If the move does nothing, does not add a tile. The defaults are the
        2048 default random tile placements."""
        if self.make_move(move_generation_func(self.board)):
            self.add_random_tile(tiles, weights)
        else:
            #self.add_random_tile(tiles, weights)
            pass
        return self.game_status()

    def play_to_completion(self, move_generation_func, per_move_callback=None, tiles=(2, 4), weights=(9, 1)):
        """Given a function that takes in the current board position and returns a move, continues play
        until the game is over, returning the game status. Tiles and weights get passed to
        add_random_tile.
        
        permove_callback is an optional function that is called after every move. It is used for model-training purposes.
        """
        self.add_random_tile(tiles, weights)
        while not self.game_status():
            self.play(move_generation_func, tiles, weights)
            if not per_move_callback is None:
                per_move_callback()
        return self.game_status

    def print_boards(self):
        """Prints the boards in order of appearance in history."""
        new_g = Game(board=self.original_board)
        for spawn, move in zip(self.spawns, self.history):
            new_g.update_history([spawn], [move])
            print(new_g)
            print('\n' + '-' * 10)

    def __write_data(self, outfile):
        outfile.write("{}\n".format(self.board.width))
        outfile.write("{}\n".format(self.board.height))
        outfile.write("{}\n".format(self.goal))
        outfile.write(' '.join(map(str, list(self.original_board.flatten()))))
        outfile.write('\n')
        for i in range(len(self.history)):
            curr_spawn = self.spawns[i]
            curr_move = self.history[i]
            outfile.write("{}-{} {}\n".format(*curr_spawn[0], curr_spawn[1]))
            outfile.write("{}\n".format(curr_move))

        # Add the final spawn after the last move made, which gets cut off, because len(self.spawns) = len(self.history) + 1
        outfile.write("{}-{} {}\n".format(*self.spawns[-1][0], self.spawns[-1][1]))

    def save(self, filename):
        """Saves this game to a file as a newline-separated list of WASD with info at the beginning.
        """
        with open(filename, 'w+') as outfile:
            self.__write_data(outfile)

    def append(self, filename, sep='#'):
        """Appends this game to an already-existing file, adding `sep` in between games"""
        if not os.path.isfile(filename):
            self.save(filename)
            return

        with open(filename, 'a') as outfile:
            outfile.write(sep)
            self.__write_data(outfile)

    @staticmethod
    def open(filename):
        """Generates a Game from the given filename."""
        with open(filename, 'r') as infile:
            lines = list(infile)
            width = int(lines[0])
            height = int(lines[1])
            goal = int(lines[2])
            tiles = [float(x) for x in lines[3].strip().split(' ')]
            tiles = np.array(tiles).reshape(height, width)
            g = Game(width, height, tiles, None, goal)
            spawns = []
            moves = []
            for i in range(4, len(lines), 2):
                spawn_line = lines[i]
                if i < len(lines) - 1:
                    move_line = lines[i+1]
                spawn_pos, spawn_tile = spawn_line.split(' ')
                spawn_x, spawn_y = spawn_pos.split('-')
                spawns.append(((int(spawn_x), int(spawn_y)), int(spawn_tile)))
                if i < len(lines) - 1:
                    moves.append(int(move_line))

            g.update_history(spawns, moves)

        return g

    @staticmethod
    def open_from_text(text):
        """Generates a Game object from the given string"""
        lines = text.split('\n')
        width = int(lines[0])
        height = int(lines[1])
        goal = int(lines[2])
        tiles = [float(x) for x in lines[3].strip().split(' ')]
        tiles = np.array(tiles).reshape(height, width)
        g = Game(width, height, tiles, None, goal)
        spawns = []
        moves = []
        for i in range(4, len(lines), 2):
            spawn_line = lines[i]
            if i < len(lines) - 1:
                move_line = lines[i+1]
            spawn_pos, spawn_tile = spawn_line.split(' ')
            spawn_x, spawn_y = spawn_pos.split('-')
            spawns.append(((int(spawn_x), int(spawn_y)), int(spawn_tile)))
            if i < len(lines) - 1:
                moves.append(int(move_line))

        g.update_history(spawns, moves)

        return g

    @staticmethod
    def open_batch(filename, sep='#'):
        with open(filename, 'r') as infile:
            
            sections = [x.strip() for x in ''.join(list(infile)).split(sep)]
            g = [Game.open_from_text(s) for s in sections]

        return g


def input_player(board):
    return "WDSA".index(input("The board is \n{}\nWhat would you like to do? ".format(board)))


def random_play(board):
    return np.random.choice(4)
