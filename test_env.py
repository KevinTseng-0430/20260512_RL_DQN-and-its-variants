from Gridworld import Gridworld

env = Gridworld(size=4, mode="static")

print(env.display())
print("Reward:", env.reward())

env.makeMove("l")
print(env.display())
print("Reward:", env.reward())