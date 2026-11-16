
import numpy as np

gamma = 0.9
theta = 0.001

# Three states: Start, Near Goal, Goal
states = [0, 1, 2]
actions = [0, 1]  # 0 = Move Forward, 1 = Stay

# Transition model: next state and reward
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

            # TODO 1: Calculate action value
            action_value = reward + gamma * V[next_state]

            # TODO 2: Append action value
            action_values.append(action_value)

        # TODO 3: Update state value
        V[s] = max(action_values)

        # TODO 4: Calculate change in value
        delta = max(delta, abs(old_value - V[s]))

    # TODO 5: Check convergence
    if delta < theta:
        break

print("Optimal State Values:", V)

# Identify optimal action
for s in [0, 1]:
    values = []

    for a in actions:
        next_state, reward = P[(s, a)]
        values.append(reward + gamma * V[next_state])

    best_action = actions[np.argmax(values)]

    if best_action == 0:
        print("State", s, "Optimal Action: Move Forward")
    else:
        print("State", s, "Optimal Action: Stay")
