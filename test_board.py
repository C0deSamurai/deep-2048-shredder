import unittest

from board import Board


class TestBoard(unittest.TestCase):

    def setUp(self):
        self.b = Board()
    
    def test_default_board_generation(self):
        self.assertTupleEqual((4, 4), self.b.board.shape)
        self.assertTupleEqual((4, 4), (self.b.width, self.b.height))

    def test_custom_board_generation(self):
        for shape in ((3, 2), (2, 9), (11, 10)):
            b = Board(*shape, board=None)
            self.assertTupleEqual(shape, (b.width, b.height))
            self.assertTupleEqual(tuple(reversed(shape)), b.board.shape)

    def test_rotation(self):
        self.b.set_tile(0, 0, 2)
        self.b.set_tile(0, 1, 4)
        self.b.set_tile(1, 2, 8)
        self.b.set_tile(3, 2, 2048)

        # board now looks like:
        # - - - -
        # - 8 - 2048
        # 4 - - -
        # 2 - - -

        old_board = self.b.board
        
        self.b.rotate(1)
        self.b.rotate(3)

        self.assertEqual(repr(old_board), repr(self.b.board))

        self.b.rotate(-1)
        self.b.rotate(2)
        self.b.rotate(3)
        self.b.rotate(-2)
        self.b.rotate(-2)

        self.assertEqual(repr(old_board), repr(self.b.board))

        self.b.rotate(1)
        self.b.rotate(-3)
        self.b.rotate(18)

        self.assertEqual(repr(old_board), repr(self.b.board))


if __name__ == "__main__":
    unittest.main()
