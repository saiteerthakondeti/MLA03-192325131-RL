
import numpy as np
import matplotlib.pyplot as plt
import random

# =====================================================
# TASK 4: SARSA, Q-LEARNING AND DEEP Q-NETWORK
# Autonomous Warehouse Robot Navigation
# =====================================================

random.seed(42)
np.random.seed(42)

# 1. PARAMETERS
ROWS = 5
COLS = 5

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

MOVES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

START = (0, 0)
GOAL = (4, 4)

WALLS = [
    (1, 1),
    (2, 1),
    (3, 3)
]

GAMMA = 0.9
ALPHA = 0.2
EPSILON = 0.1
EPISODES = 2000
MAX_STEPS = 100


# =====================================================
# 2. WAREHOUSE ENVIRONMENT
# =====================================================

def state_number(position):
    return position[0] * COLS + position[1]


def take_action(position, action):

    dr, dc = MOVES[action]

    new_position = (
        position[0] + dr,
        position[1] + dc
    )

    # Boundary and obstacle checking
    if (
        new_position[0] < 0 or
        new_position[0] >= ROWS or
        new_position[1] < 0 or
        new_position[1] >= COLS or
        new_position in WALLS
    ):
        return position, -5, False

    if new_position == GOAL:
        return new_position, 20, True

    return new_position, -1, False


# =====================================================
# 3. DISPLAY WAREHOUSE
# =====================================================

def display_grid():

    print("\nWAREHOUSE GRID\n")

    for r in range(ROWS):

        for c in range(COLS):

            pos = (r, c)

            if pos == START:
                print("S", end=" ")

            elif pos == GOAL:
                print("G", end=" ")

            elif pos in WALLS:
                print("#", end=" ")

            else:
                print(".", end=" ")

        print()


# =====================================================
# 4. EPSILON-GREEDY ACTION SELECTION
# =====================================================

def choose_action(Q, state):

    if random.random() < EPSILON:
        return random.randint(0, 3)

    return int(np.argmax(Q[state]))


# =====================================================
# 5. VALUE ITERATION
# =====================================================

def value_iteration():

    V = np.zeros(ROWS * COLS)

    theta = 0.001

    while True:

        delta = 0

        for r in range(ROWS):
            for c in range(COLS):

                position = (r, c)

                if position == GOAL or position in WALLS:
                    continue

                state = state_number(position)
                old_value = V[state]

                action_values = []

                for action in range(4):

                    next_pos, reward, done = take_action(
                        position, action
                    )

                    next_state = state_number(next_pos)

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

        if delta < theta:
            break

    return V


# =====================================================
# 6. SARSA ALGORITHM - ON POLICY
# =====================================================

def train_sarsa():

    Q = np.zeros((25, 4))

    rewards_history = []
    success_count = 0

    for episode in range(EPISODES):

        position = START
        state = state_number(position)

        action = choose_action(Q, state)

        total_reward = 0

        for step in range(MAX_STEPS):

            next_pos, reward, done = take_action(
                position, action
            )

            next_state = state_number(next_pos)

            if done:
                target = reward
                next_action = None

            else:
                next_action = choose_action(Q, next_state)

                target = (
                    reward +
                    GAMMA * Q[next_state, next_action]
                )

            # SARSA update
            Q[state, action] += ALPHA * (
                target - Q[state, action]
            )

            total_reward += reward

            if done:
                success_count += 1
                break

            position = next_pos
            state = next_state
            action = next_action

        rewards_history.append(total_reward)

    success_rate = success_count / EPISODES * 100

    return Q, rewards_history, success_rate


# =====================================================
# 7. Q-LEARNING ALGORITHM - OFF POLICY
# =====================================================

def train_q_learning():

    Q = np.zeros((25, 4))

    rewards_history = []
    success_count = 0

    for episode in range(EPISODES):

        position = START
        state = state_number(position)

        total_reward = 0

        for step in range(MAX_STEPS):

            action = choose_action(Q, state)

            next_pos, reward, done = take_action(
                position, action
            )

            next_state = state_number(next_pos)

            if done:
                target = reward

            else:
                target = (
                    reward +
                    GAMMA * np.max(Q[next_state])
                )

            # Q-Learning update
            Q[state, action] += ALPHA * (
                target - Q[state, action]
            )

            total_reward += reward

            if done:
                success_count += 1
                break

            position = next_pos
            state = next_state

        rewards_history.append(total_reward)

    success_rate = success_count / EPISODES * 100

    return Q, rewards_history, success_rate


# =====================================================
# 8. DISPLAY LEARNED NAVIGATION PATH
# =====================================================

def display_path(Q, algorithm_name):

    position = START
    path = [position]
    visited = set()

    print("\n", algorithm_name, "NAVIGATION PATH")

    for step in range(MAX_STEPS):

        state = state_number(position)

        if state in visited:
            print("Repeated state detected.")
            break

        visited.add(state)

        action = int(np.argmax(Q[state]))

        next_pos, reward, done = take_action(
            position, action
        )

        if next_pos == position:
            break

        position = next_pos
        path.append(position)

        if done:
            break

    print("Path:", path)
    print("Total Steps:", len(path) - 1)

    if position == GOAL:
        print("Status: Goal Reached")
    else:
        print("Status: Goal Not Reached")

    # Draw path
    grid = np.zeros((ROWS, COLS))

    for r, c in WALLS:
        grid[r, c] = -1

    for r, c in path:
        if (r, c) not in [START, GOAL]:
            grid[r, c] = 1

    plt.figure(figsize=(6, 5))
    plt.imshow(grid, cmap="coolwarm")

    for r in range(ROWS):
        for c in range(COLS):

            label = "."

            if (r, c) == START:
                label = "S"
            elif (r, c) == GOAL:
                label = "G"
            elif (r, c) in WALLS:
                label = "#"
            elif (r, c) in path:
                label = "*"

            plt.text(c, r, label,
                     ha="center", va="center")

    plt.title(algorithm_name + " Navigation Path")
    plt.xticks(range(COLS))
    plt.yticks(range(ROWS))
    plt.grid()
    plt.show()

    return len(path) - 1, position == GOAL


# =====================================================
# 9. NUMERICAL SARSA CALCULATION
# =====================================================

def numerical_sarsa():

    q_current = 5
    reward = 4
    alpha = 0.2
    gamma = 0.9
    next_action_value = 6

    updated_q = q_current + alpha * (
        reward + gamma * next_action_value - q_current
    )

    print("\nNUMERICAL SARSA CALCULATION")

    print("Q(s,a) =", q_current)
    print("Reward =", reward)
    print("Learning Rate =", alpha)
    print("Discount Factor =", gamma)
    print("Next Action Value =", next_action_value)

    print("Updated SARSA Q-value =", round(updated_q, 2))


# =====================================================
# 10. NUMERICAL Q-LEARNING CALCULATION
# =====================================================

def numerical_qlearning():

    q_current = 5
    reward = 4
    alpha = 0.2
    gamma = 0.9
    max_next_value = 8

    updated_q = q_current + alpha * (
        reward + gamma * max_next_value - q_current
    )

    print("\nNUMERICAL Q-LEARNING CALCULATION")

    print("Q(s,a) =", q_current)
    print("Reward =", reward)
    print("Learning Rate =", alpha)
    print("Discount Factor =", gamma)
    print("Maximum Next Q-value =", max_next_value)

    print("Updated Q-Learning Q-value =", round(updated_q, 2))


# =====================================================
# 11. DQN CONCEPTS
# =====================================================

def dqn_explanation():

    print("\nDEEP Q-NETWORK CONCEPTS")

    print("1. DQN: Neural network approximates Q-values.")

    print("2. Experience Replay: Stores and samples past experiences.")

    print("3. Target Network: Provides stable target Q-values.")

    print("4. DDQN: Reduces overestimation of Q-values.")

    print("5. Dueling DQN: Separates state value and action advantage.")

    print("6. Prioritised Experience Replay: Samples important experiences more often.")


# =====================================================
# 12. PERFORMANCE GRAPH
# =====================================================

def performance_graph(sarsa_rewards, q_rewards):

    plt.figure(figsize=(10, 5))

    plt.plot(sarsa_rewards, label="SARSA")
    plt.plot(q_rewards, label="Q-Learning")

    plt.xlabel("Episodes")
    plt.ylabel("Episode Reward")

    plt.title("SARSA vs Q-Learning Performance")

    plt.legend()
    plt.grid()

    plt.show()


# =====================================================
# 13. MAIN PROGRAM
# =====================================================

if __name__ == "__main__":

    print("=" * 55)
    print("TASK 4: SARSA, Q-LEARNING AND DQN")
    print("WAREHOUSE ROBOT NAVIGATION")
    print("=" * 55)

    display_grid()

    # Value Iteration
    print("\nRunning Value Iteration...")

    V = value_iteration()

    print("Value Iteration Completed.")
    print("Start State Value:", round(V[state_number(START)], 3))

    # SARSA
    print("\nTraining SARSA...")

    sarsa_Q, sarsa_rewards, sarsa_success = train_sarsa()

    print("SARSA Training Completed.")
    print("SARSA Success Rate:", round(sarsa_success, 2), "%")

    # Q-Learning
    print("\nTraining Q-Learning...")

    q_Q, q_rewards, q_success = train_q_learning()

    print("Q-Learning Training Completed.")
    print("Q-Learning Success Rate:", round(q_success, 2), "%")

    # Navigation paths
    sarsa_steps, sarsa_reached = display_path(
        sarsa_Q, "SARSA"
    )

    q_steps, q_reached = display_path(
        q_Q, "Q-Learning"
    )

    # Numerical calculations
    numerical_sarsa()
    numerical_qlearning()

    # DQN concepts
    dqn_explanation()

    # Experimental comparison
    print("\nFINAL EXPERIMENTAL COMPARISON")

    print("-" * 65)

    print(
        "{:<20} {:<18} {:<18}".format(
            "Algorithm", "Success Rate", "Average Reward"
        )
    )

    print("-" * 65)

    print(
        "{:<20} {:<18.2f} {:<18.2f}".format(
            "SARSA",
            sarsa_success,
            np.mean(sarsa_rewards)
        )
    )

    print(
        "{:<20} {:<18.2f} {:<18.2f}".format(
            "Q-Learning",
            q_success,
            np.mean(q_rewards)
        )
    )

    print("-" * 65)

    # Performance graph
    performance_graph(sarsa_rewards, q_rewards)

    print("\nTASK 4 PROGRAM COMPLETED.")
