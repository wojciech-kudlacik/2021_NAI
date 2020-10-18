# Source: https://www.codingame.com/training/easy/power-of-thor-episode-1
# Author: Wojciech Kudłacik

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
# ---
# Hint: You can use the debug stream to print initialTX and initialTY, if Thor seems not follow your orders.

# light_x: the X position of the light of power
# light_y: the Y position of the light of power
# initial_tx: Thor's starting X position
# initial_ty: Thor's starting Y position
light_x, light_y, initial_tx, initial_ty = [int(i) for i in input().split()]
current_tx, current_ty = initial_tx, initial_ty

# game loop
while True:
    remaining_turns = int(input())  # The remaining amount of turns Thor can move. Do not remove this line.
    direction_x = direction_y = ""

    if current_ty > light_y:
        direction_y += "N"
        current_ty -= 1
    elif current_ty < light_y:
        direction_y += "S"
        current_ty += 1
    if current_tx > light_x:
        direction_x += "W"
        current_tx += 1
    elif current_tx < light_x:
        direction_x += "E"
        current_tx -= 1

    # A single line providing the move to be made: N NE E SE S SW W or NW
    # Directions are written in YX format because Y axis is N/S and X axis is W/E
    print(direction_y + direction_x)

