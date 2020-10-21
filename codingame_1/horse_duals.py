# Source: https://www.codingame.com/training/easy/horse-racing-duals
# Author: Wojciech Kudłacik
# Important tutorial used: https://www.geeksforgeeks.org/find-minimum-difference-pair/

# data needed
n = int(input())  # number of horses
horses_strength = []  # list of all horses' stregth indicators
for i in range(n):
    pi = int(input())  # strength of each horse
    horses_strength.append(pi)

# calculate minimal difference between two numbers in a list
sorted_horses = sorted(horses_strength)  # sorted array of horses' strength
difference = abs(sorted_horses[1] - sorted_horses[0])  # initial difference between two first elements

for i in range(n-1):
    if sorted_horses[i+1] - sorted_horses[i] < difference:
        difference = sorted_horses[i+1] - sorted_horses[i]

# result
print(difference)

