
import numpy as np

gamma = 0.9
theta = 0.001

# Three states
states = [0, 1, 2]

# Actions: 0 = Move Forward, 1 = Stay
actions = [0, 1]

# Transition model
P = {
    (0, 0): (1, -1),
    (0, 1): (0, -2),
    (1, 0): (2, 10),
    (1, 1): (1, -2)
}

# Initialize state values
V = np.zeros(len(states))

# Value Iteration
while True:
    delta = 0

    for s in [0, 1]:
        old_value = V[s]
        action_values = []

        for a in actions:
            next_state, reward = P[(s, a)]

            action_value = reward + gamma * V[next_state]
            action_values.append(action_value)

        V[s] = max(action_values)

        delta = max(delta, abs(old_value - V[s]))

    if delta < theta:
        break

print("Optimal State Values:", V)

# Find optimal actions
for s in [0, 1]:
    action_values = []

    for a in actions:
        next_state, reward = P[(s, a)]
        action_values.append(reward + gamma * V[next_state])

    print("State:", s, "Optimal Action:", actions[np.argmax(action_values)])
