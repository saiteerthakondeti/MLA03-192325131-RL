
import numpy as np

# Parameters
gamma = 0.9
episodes = 1000

# Five warehouse states
# 0 = Start
# 1 = Shelf A
# 2 = Shelf B
# 3 = Packing Station
# 4 = Obstacle area

rewards = {
    0: -1,
    1: 5,
    2: 8,
    3: 20,
    4: -10
}

# Initialize state values
V = np.zeros(5)
returns = {s: [] for s in range(5)}

# Generate episodes
for episode in range(episodes):

    trajectory = [0, 1, 2, 3]
    episode_rewards = [-1, 5, 8, 20]

    G = 0
    episode_returns = []

    for reward in reversed(episode_rewards):
        G = reward + gamma * G
        episode_returns.insert(0, G)

    # First-Visit Monte Carlo
    visited = set()

    for state, G in zip(trajectory, episode_returns):

        if state not in visited:
            returns[state].append(G)
            V[state] = np.mean(returns[state])
            visited.add(state)

print("Monte Carlo State Values:")

for state in range(5):
    print("State", state, "Value:", round(V[state], 2))

print("\nMonte Carlo Training Completed!")
print("Total Episodes:", episodes)
