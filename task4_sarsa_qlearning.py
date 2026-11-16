
import numpy as np
import random

# --------------------------------
# NUMERICAL CALCULATIONS
# --------------------------------

Q = 5
reward = 4
alpha = 0.2
gamma = 0.9

# SARSA
next_sarsa = 6

sarsa_value = Q + alpha * (
    reward + gamma * next_sarsa - Q
)

# Q-Learning
max_next_q = 8

qlearning_value = Q + alpha * (
    reward + gamma * max_next_q - Q
)

print("NUMERICAL CALCULATIONS")
print("Updated SARSA Q-value:", round(sarsa_value, 2))
print("Updated Q-Learning Q-value:", round(qlearning_value, 2))


# --------------------------------
# WAREHOUSE ENVIRONMENT
# --------------------------------

ROWS = 5
COLS = 5

start = (0, 0)
goal = (4, 4)

obstacles = {
    (1, 1),
    (2, 1),
    (3, 3)
}

actions = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

def step(state, action):
    if state == goal:
        return goal, 0

    dr, dc = actions[action]
    nr = state[0] + dr
    nc = state[1] + dc

    next_state = (nr, nc)

    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return state, -10

    if next_state in obstacles:
        return state, -10

    if next_state == goal:
        return goal, 100

    return next_state, -1


def choose_action(Q_table, state, epsilon):
    if random.random() < epsilon:
        return random.choice(list(actions.keys()))

    return int(np.argmax(Q_table[state[0], state[1]]))


# --------------------------------
# SARSA ALGORITHM
# --------------------------------

def train_sarsa(episodes=2000):
    Q_table = np.zeros((ROWS, COLS, 4))

    for episode in range(episodes):
        state = start
        action = choose_action(Q_table, state, 0.1)

        for _ in range(100):
            next_state, reward = step(state, action)

            if next_state == goal:
                target = reward
                Q_table[state[0], state[1], action] += (
                    alpha * (target - Q_table[state[0], state[1], action])
                )
                break

            next_action = choose_action(Q_table, next_state, 0.1)

            target = reward + gamma * Q_table[
                next_state[0], next_state[1], next_action
            ]

            Q_table[state[0], state[1], action] += alpha * (
                target - Q_table[state[0], state[1], action]
            )

            state = next_state
            action = next_action

    return Q_table


# --------------------------------
# Q-LEARNING ALGORITHM
# --------------------------------

def train_qlearning(episodes=2000):
    Q_table = np.zeros((ROWS, COLS, 4))

    for episode in range(episodes):
        state = start

        for _ in range(100):
            action = choose_action(Q_table, state, 0.1)
            next_state, reward = step(state, action)

            target = reward + gamma * np.max(
                Q_table[next_state[0], next_state[1]]
            )

            if next_state == goal:
                target = reward

            Q_table[state[0], state[1], action] += alpha * (
                target - Q_table[state[0], state[1], action]
            )

            state = next_state

            if state == goal:
                break

    return Q_table


# --------------------------------
# TRAIN AND DISPLAY RESULTS
# --------------------------------

random.seed(42)
np.random.seed(42)

sarsa_Q = train_sarsa()
qlearning_Q = train_qlearning()

print("\nSARSA TRAINING COMPLETED")
print("Q-Learning TRAINING COMPLETED")


# --------------------------------
# DISPLAY LEARNED PATH
# --------------------------------

def display_path(Q_table, algorithm_name):
    state = start
    path = [state]

    for _ in range(ROWS * COLS * 2):
        if state == goal:
            break

        action = int(np.argmax(Q_table[state[0], state[1]]))
        next_state, reward = step(state, action)

        if next_state == state:
            print(algorithm_name, "reached a blocked move.")
            break

        state = next_state
        path.append(state)

    print("\n", algorithm_name, "PATH:")
    print(path)

    if state == goal:
        print("Destination reached!")
    else:
        print("Destination not reached.")


display_path(sarsa_Q, "SARSA")
display_path(qlearning_Q, "Q-Learning")
