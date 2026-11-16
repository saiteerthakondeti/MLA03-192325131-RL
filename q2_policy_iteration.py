
import numpy as np

gamma = 0.9
theta = 0.001

states = [0, 1, 2]
actions = [0, 1]

# Transition model
P = {
    (0, 0): (1, -1),
    (0, 1): (0, -2),
    (1, 0): (2, 10),
    (1, 1): (1, -2)
}

# Initialize values and policy
V = np.zeros(3)
policy = np.zeros(3, dtype=int)

# Policy Iteration
while True:

    # Policy Evaluation
    while True:
        delta = 0

        for s in [0, 1]:
            old_value = V[s]

            next_state, reward = P[(s, policy[s])]
            V[s] = reward + gamma * V[next_state]

            delta = max(delta, abs(old_value - V[s]))

        if delta < theta:
            break

    # Policy Improvement
    policy_stable = True

    for s in [0, 1]:
        old_action = policy[s]

        action_values = []

        for a in actions:
            next_state, reward = P[(s, a)]
            value = reward + gamma * V[next_state]
            action_values.append(value)

        policy[s] = np.argmax(action_values)

        if old_action != policy[s]:
            policy_stable = False

    if policy_stable:
        break

print("Optimal State Values:", V)
print("Optimal Policy:")

for s in [0, 1]:
    print("State:", s, "Action:", actions[policy[s]])
