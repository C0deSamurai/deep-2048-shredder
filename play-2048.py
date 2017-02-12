"""Plays a simple text game of 2048."""

from board import Board

b = Board()

b.add_random_tile()

while not b.game_status():
    print(b)
    direction = input()
    while direction not in "WASD":
        direction = input("One of WASD: ")
    b.make_move("WDSA".index(direction))
    b.add_random_tile()

print(b.game_status())
