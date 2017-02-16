"""This class represents the game state of a 2048 game in progress, intentionally made as general as
possible to accomodate variations on the game."""


from random import choice

import numpy as np


class Board:
    """A 2048 game board with tiles on it."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, width=4, height=4, board=None):
        """Initializes a 2048 game board of the given width and height. If board is not None, it is
        interpreted as a list of rows (of shape (height, width)) that contain numerical values
        (which should probably be powers of 2, but perhaps wouldn't need to be), interspersed with
        zeroes to denote empty squares.

        """

        if width <= 1 or height <= 1:
            raise ValueError("Need 2+ rows and columns, not shape ({}, {})".format(height, width))

        if board is None:
            self.board = np.zeros((height, width))
        else:
            self.board = np.array(board)
            if board.shape != (height, width):
                raise ValueError("Expected board of shape" +
                                 " ({}, {}), not {}".format(height, width, board.shape))

        self.height, self.width = self.board.shape

    def __repr__(self):
        return "Board({}, {}, \n{})".format(self.width, self.height, repr(self.board))

    def __str__(self):
        return '\n'.join(['\t'.join(map(str, row)) for row in self.board])
        
    def game_status(self, goal=2048):
        """Determines if the game has ended. If the game is continuing, returns 0. Otherwise,
        returns -1 for a loss (defined as no valid moves) and 1 for a win (defined as any position
        with the goal tile or a larger one on the board)."""

        if (self.board >= goal).any():  # we have a winner!
            return 1

        # if the board is filled, you can only keep playing if you have a match to make
        if (self.board != 0).all():
            # take differences and check for zeros
            if (np.diff(self.board) == 0).any() or (np.diff(self.board, axis=0) == 0).any():
                # we have a difference of zero, so there is a possible match
                return 0  # the game continues
            else:  # the game ends, no moves: you lose
                return -1
        else:
            return 0  # there are empty squares, so there must be possible moves

    def get_tile(self, x, y):
        """Returns the tile at (x, y) (where 0 means nothing is there), and the coordinate system
        has (0, 0) at the lower-left corner."""
        return self.board[self.height-y-1][self.width-x-1]

    def set_tile(self, x, y, num=2, force_replace=False):
        """Adds a tile with the given numerical value at the given x-y coordinate, in a coordinate system
        where (0, 0) is the lower-left corner and x is horizontal. If force_replace is True, no
        error will be raised if the tile is already occupied: if it is False, a ValueError will
        occur.

        """
        if self.get_tile(x, y) != 0 and not force_replace:
            raise ValueError("Tried to add a tile to a non-empty square!")
        else:
            self.board[self.height-y-1][self.width-x-1] = num

    def can_combine(self, tile1, tile2):
        """Determines if tile1 and tile2 can match or merge together. In standard 2048, this is
        simply equality, with an added check so that 0 never combines."""
        if tile1 == 0 or tile2 == 0:
            return False

        return tile1 == tile2

    def combine(self, tile1, tile2):
        """Adds together two tiles and returns the numerical value of the new tile. Override this to
        change how tiles are added together. Uses the can_combine method to check for the
        possibility to match: in standard 2048, that is simply equality, although it will never let
        0 combine."""
        if self.can_combine(tile1, tile2):
            return tile1 + tile2
        else:
            raise ValueError("{} and {} cannot combine!".format(tile1, tile2))

    def rotate(self, num_clockwise_turns=1):
        """Rotates the board with the given amount of clockwise turns, so 4 turns does nothing and 3
        turns is one counter-clockwise turn. Example turning once:

        -  -  2  4        -  -  -  -
        -  2  16 8        -  4  -  2
        -  -  2  4   ==>  2  2  16 2
        -  4  2  32       32 4  8  4

        If necessary, width and height will change to reflect the new board.

        Additionally, this method accepts negative numbers and will do the specified amount of
        counter-clockwise turns if the number is negative.
        """
        # if you're interested, the way to do this is by transposing and then flipping rows
        # but numpy already has this, so why bother?

        # this method rotates CCW, so do the negative number of turns..
        self.board = np.rot90(self.board, -num_clockwise_turns)
        self.height, self.width = self.board.shape

    def __collapse_row(self, row):
        """Collapses the numpy array row in-place leftwards according to the logic of 2048: e.g.,
        - 16 4 4  ==>  16 8 - -
        Returns None."""

        # the base case of nothing: important, otherwise the below will loop forever
        if np.count_nonzero(row) == 0:  # all blank
            return row

        r = list(row)
        # if there is space to move leftwards, does so and adds space to the end
        while r[0] == 0:
            r = r[1:] + [r[0]]

        # now, go through and collapse any blank space between two tiles
        for i in range(1, len(r) - 1):

            # if there is some filled square that should collapse into this spot on the right
            # keep collapsing blank squares until this "run" is over
            while r[i] == 0 and not all([x == 0 for x in r[i+1:]]):
                r = r[:i] + r[i+1:] + [r[i]]  # shuffle this space to the end, keep length constant

        # now, the fun part: actually collapsing tiles, before we remove empty squares again
        # because can_combine rejects blank squares, no need to worry about them here
        for i in range(len(r) - 1):
            if self.can_combine(r[i], r[i+1]):
                # combine and then erase one of the tiles
                r[i] = self.combine(r[i], r[i+1])
                r[i+1] = 0

        # now just remove every zero and add back zeroes at the end as padding
        old_l = len(r)
        r = [x for x in r if x != 0]
        r += [0 for i in range(old_l - len(r))]

        return np.array(r)

    def __move_left(self):
        """Collapses the board leftwards. This method is private and outside users should use
        make_move, which just rotates the board, does this, and rotates back. Returns None."""
        self.board = np.apply_along_axis(self.__collapse_row, 1, self.board)
        
    def make_move(self, direction):
        """Shifts the board in the given direction, and merges any tiles that can be merged in the given
        direction. May not do anything. Direction is a number 0-3 representing up, right, down, and
        left respectively (clockwise), which can be replaced by the constant variables UP, DOWN,
        LEFT, and RIGHT from this class. Returns 1 if the given move does something, and 0
        otherwise.

        """

        self.old_board = np.copy(self.board)
        # rotate until what was the desired direction is facing left, collapse leftwards, and then
        # rotate back
        self.rotate(3 - direction)  # clockwise numbering jells nicely with rotation

        self.__move_left()  # do the actual work of collapsing
        self.rotate(direction - 3)  # undo whatever was done previously

        if (self.board == self.old_board).all():  # same board
            return 0
        else:
            return 1

    def add_random_tile(self, tiles=(2, 4), weights=(9, 1)):
        """Adds a tile in a randomly chosen unoccupied position according to the given weighted selection of
        tiles. If the board is full, does nothing. Returns None.

        The default is how the original 2048 (gabrielcirulli.github.io/2048/) does it:
        there's a 90% of getting a 2, but a 10% chance of getting a 4.

        """
        unoccupied_slots = []
        for x in range(self.width):
            for y in range(self.height):
                if self.get_tile(x, y) == 0:
                    unoccupied_slots.append((x, y))

        if not unoccupied_slots:  # nowhere to add
            return None

        # note: np.random.choice cannot make choices from tuples of tuples
        # so we have to import the standard library's random.choice
        slot_to_add = choice(unoccupied_slots)

        normed_weights = [weight / sum(weights) for weight in weights]
        # this one requires weighted sampling, so we use np.random.choice
        # because tiles is just a list of integers
        val_to_add = np.random.choice(tiles, p=normed_weights)

        self.set_tile(*slot_to_add, val_to_add)
