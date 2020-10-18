# Source: https://www.codingame.com/training/easy/the-descent
# Author: Wojciech Kudłacik

# The while loop represents the game.
# Each iteration represents a turn of the game
# where you are given inputs (the heights of the mountains)
# and where you have to print an output (the index of the mountain to fire on)
# The inputs you are given are automatically updated according to your last actions.

# game loop
while True:
    mountain_i = 0
    max_h = 0

    for i in range(8):
        mountain_h = int(input())  # represents the height of one mountain.

        if mountain_h > max_h:
            max_h = mountain_h
            mountain_i = i

    # The index of the mountain to fire on.
    print(mountain_i)