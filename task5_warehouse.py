
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. WAREHOUSE ENVIRONMENT
# ==========================================

ROWS = 5
COLS = 5

START = (0, 0)
GOAL = (4, 4)

OBSTACLES = {
    (1, 1),
    (2, 1),
    (3, 3)
}

ACTIONS = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

GAMMA = 0.9
ALPHA = 0.1
THETA = 0.001

STATES = [
    (r, c)
    for r in range(ROWS)
    for c in range(COLS)
    if (r, c) not in OBSTACLES
]


# ==========================================
# 2. ENVIRONMENT TRANSITION
# ==========================================

def step(state, action):

    if state == GOAL:
        return GOAL, 0, True

    dr, dc = ACTIONS[action]

    nr = state[0] + dr
    nc = state[1] + dc

    next_state = (nr, nc)

    # Boundary penalty
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return state, -10, False

    # Obstacle penalty
    if next_state in OBSTACLES:
        return state, -10, False

    # Goal reward
    if next_state == GOAL:
        return next_state, 100, True

    # Normal movement
    return next_state, -1, False


# ==========================================
# 3. DYNAMIC PROGRAMMING - VALUE ITERATION
# ==========================================

def value_iteration():

    V = {s: 0.0 for s in STATES}

    while True:

        delta = 0

        for state in STATES:

            if state == GOAL:
                continue

            old_value = V[state]

            action_values = []

            for action in ACTIONS:

                next_state, reward, done = step(state, action)

                if done:
                    value = reward
                else:
                    value = reward + GAMMA * V[next_state]

                action_values.append(value)

            V[state] = max(action_values)

            delta = max(
                delta,
                abs(old_value - V[state])
            )

        if delta < THETA:
            break

    policy = {}

    for state in STATES:

        if state == GOAL:
            continue

        values = []

        for action in ACTIONS:

            next_state, reward, done = step(state, action)

            if done:
                value = reward
            else:
                value = reward + GAMMA * V[next_state]

            values.append(value)

        policy[state] = int(np.argmax(values))

    return V, policy


# ==========================================
# 4. SARSA ALGORITHM
# ==========================================

def choose_action(Q, state, epsilon):

    if random.random() < epsilon:
        return random.choice(list(ACTIONS.keys()))

    return int(np.argmax(Q[state[0], state[1]]))


def train_sarsa(episodes=3000):

    Q = np.zeros((ROWS, COLS, 4))

    rewards_history = []

    for episode in range(episodes):

        state = START

        epsilon = max(
            0.02,
            0.2 * (1 - episode / episodes)
        )

        action = choose_action(Q, state, epsilon)

        total_reward = 0

        for _ in range(100):

            next_state, reward, done = step(state, action)

            total_reward += reward

            if done:
                target = reward

                Q[state[0], state[1], action] += ALPHA * (
                    target - Q[state[0], state[1], action]
                )

                break

            next_action = choose_action(
                Q, next_state, epsilon
            )

            target = reward + GAMMA * Q[
                next_state[0],
                next_state[1],
                next_action
            ]

            Q[state[0], state[1], action] += ALPHA * (
                target - Q[state[0], state[1], action]
            )

            state = next_state
            action = next_action

        rewards_history.append(total_reward)

    return Q, rewards_history


# ==========================================
# 5. Q-LEARNING ALGORITHM
# ==========================================

def train_qlearning(episodes=3000):

    Q = np.zeros((ROWS, COLS, 4))

    rewards_history = []

    for episode in range(episodes):

        state = START

        epsilon = max(
            0.02,
            0.2 * (1 - episode / episodes)
        )

        total_reward = 0

        for _ in range(100):

            action = choose_action(Q, state, epsilon)

            next_state, reward, done = step(state, action)

            total_reward += reward

            if done:
                target = reward
            else:
                target = reward + GAMMA * np.max(
                    Q[next_state[0], next_state[1]]
                )

            Q[state[0], state[1], action] += ALPHA * (
                target - Q[state[0], state[1], action]
            )

            state = next_state

            if done:
                break

        rewards_history.append(total_reward)

    return Q, rewards_history


# ==========================================
# 6. EXTRACT NAVIGATION PATH
# ==========================================

def get_path(policy):

    state = START
    path = [state]
    visited = {state}

    for _ in range(ROWS * COLS * 2):

        if state == GOAL:
            break

        action = policy[state]

        next_state, reward, done = step(state, action)

        if next_state == state or next_state in visited:
            break

        state = next_state
        path.append(state)
        visited.add(state)

        if done:
            break

    return path


def get_q_policy(Q):

    policy = {}

    for state in STATES:

        if state != GOAL:
            policy[state] = int(
                np.argmax(Q[state[0], state[1]])
            )

    return policy


# ==========================================
# 7. EVALUATE ALGORITHMS
# ==========================================

def evaluate_path(path):

    total_reward = 0
    steps = 0
    success = path[-1] == GOAL

    for i in range(len(path) - 1):

        state = path[i]
        next_state = path[i + 1]

        for action in ACTIONS:

            result, reward, done = step(state, action)

            if result == next_state:
                total_reward += reward
                steps += 1
                break

    return total_reward, steps, success


# ==========================================
# 8. MAIN PROGRAM
# ==========================================

random.seed(42)
np.random.seed(42)

print("TRAINING WAREHOUSE NAVIGATION SYSTEM")

# Value Iteration
V, dp_policy = value_iteration()

# SARSA
sarsa_Q, sarsa_rewards = train_sarsa()

# Q-Learning
qlearning_Q, qlearning_rewards = train_qlearning()

# Extract policies
sarsa_policy = get_q_policy(sarsa_Q)
qlearning_policy = get_q_policy(qlearning_Q)

# Extract paths
dp_path = get_path(dp_policy)
sarsa_path = get_path(sarsa_policy)
qlearning_path = get_path(qlearning_policy)

# Evaluate paths
dp_metrics = evaluate_path(dp_path)
sarsa_metrics = evaluate_path(sarsa_path)
qlearning_metrics = evaluate_path(qlearning_path)

# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\nVALUE ITERATION PATH:")
print(dp_path)

print("\nSARSA PATH:")
print(sarsa_path)

print("\nQ-LEARNING PATH:")
print(qlearning_path)

print("\nPERFORMANCE COMPARISON")

print(
    f"{'Algorithm':<18}"
    f"{'Reward':<12}"
    f"{'Steps':<10}"
    f"{'Success':<12}"
)

results = {
    "Value Iteration": dp_metrics,
    "SARSA": sarsa_metrics,
    "Q-Learning": qlearning_metrics
}

for name, metrics in results.items():

    reward, steps, success = metrics

    print(
        f"{name:<18}"
        f"{reward:<12}"
        f"{steps:<10}"
        f"{str(success):<12}"
    )

# ==========================================
# 10. LEARNING CURVE
# ==========================================

def moving_average(data, window=100):

    return np.convolve(
        data,
        np.ones(window) / window,
        mode="valid"
    )


plt.figure(figsize=(9, 5))

plt.plot(
    moving_average(sarsa_rewards),
    label="SARSA"
)

plt.plot(
    moving_average(qlearning_rewards),
    label="Q-Learning"
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Learning Performance Comparison")
plt.legend()
plt.grid(True)

plt.savefig("learning_curves.png")
plt.show()


# ==========================================
# 11. PERFORMANCE GRAPH
# ==========================================

names = list(results.keys())

success_rates = [
    100 if results[name][2] else 0
    for name in names
]

steps = [
    results[name][1]
    for name in names
]

x = np.arange(len(names))

plt.figure(figsize=(9, 5))

plt.bar(x, success_rates)

plt.xticks(x, names)
plt.ylabel("Success Rate (%)")
plt.title("Warehouse Navigation Success Rate")
plt.ylim(0, 110)
plt.grid(axis="y")

plt.savefig("performance_comparison.png")
plt.show()


# ==========================================
# 12. DISPLAY WAREHOUSE
# ==========================================

def display_warehouse(path):

    grid = [["." for _ in range(COLS)] for _ in range(ROWS)]

    for r, c in OBSTACLES:
        grid[r][c] = "#"

    for r, c in path:
        grid[r][c] = "*"

    grid[START[0]][START[1]] = "S"
    grid[GOAL[0]][GOAL[1]] = "G"

    print("\nFINAL WAREHOUSE NAVIGATION PATH")

    for row in grid:
        print(" ".join(row))


display_warehouse(qlearning_path)

print("\nPROGRAM COMPLETED")
print("Graphs saved successfully.")
