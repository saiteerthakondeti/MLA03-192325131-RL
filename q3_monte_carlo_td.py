
import random
from collections import defaultdict

# ============================================================
# TASK 3: MONTE CARLO CONTROL AND TEMPORAL-DIFFERENCE LEARNING
# Autonomous Warehouse Delivery Robot
# ============================================================

# Warehouse grid: 5 rows x 5 columns
# S = Start, G = Goal, # = Obstacle
GRID = [
    ["S", ".", ".", "#", "."],
    [".", "#", ".", "#", "."],
    [".", "#", ".", ".", "."],
    [".", ".", "#", "#", "."],
    [".", ".", ".", ".", "G"]
]

START = (0, 0)
GOAL = (4, 4)

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

GAMMA = 0.9
ALPHA = 0.2
EPSILON = 0.1
EPISODES = 3000
MAX_STEPS = 100

random.seed(42)


# ============================================================
# 1. WAREHOUSE ENVIRONMENT
# ============================================================

def step(state, action):
    dr, dc = ACTIONS[action]

    new_state = (state[0] + dr, state[1] + dc)

    # Obstacle or boundary
    if (
        new_state[0] < 0 or new_state[0] >= 5 or
        new_state[1] < 0 or new_state[1] >= 5 or
        GRID[new_state[0]][new_state[1]] == "#"
    ):
        return state, -5, False

    # Successful delivery
    if new_state == GOAL:
        return new_state, 20, True

    # Normal movement cost
    return new_state, -1, False


def generate_episode(policy=None, epsilon=0.1):
    """Generate one complete robot navigation episode."""

    state = START
    episode = []

    for _ in range(MAX_STEPS):

        # Epsilon-soft action selection
        if policy is None or random.random() < epsilon:
            action = random.choice(list(ACTIONS.keys()))
        else:
            action = policy(state)

        next_state, reward, done = step(state, action)

        episode.append((state, action, reward))

        state = next_state

        if done:
            break

    return episode


# ============================================================
# 2. CALCULATE MONTE CARLO RETURNS
# ============================================================

def calculate_returns(episode):
    """Calculate discounted returns from the end of an episode."""

    G = 0
    returns = []

    for state, action, reward in reversed(episode):
        G = reward + GAMMA * G
        returns.append((state, action, G))

    returns.reverse()
    return returns


# ============================================================
# 3. FIRST-VISIT MONTE CARLO PREDICTION
# ============================================================

def first_visit_mc(episodes):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    values = defaultdict(float)

    for episode in episodes:
        returns = calculate_returns(episode)
        visited = set()

        for state, action, G in returns:
            if state not in visited:
                returns_sum[state] += G
                returns_count[state] += 1
                visited.add(state)

    for state in returns_sum:
        values[state] = returns_sum[state] / returns_count[state]

    return values


# ============================================================
# 4. EVERY-VISIT MONTE CARLO PREDICTION
# ============================================================

def every_visit_mc(episodes):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    values = defaultdict(float)

    for episode in episodes:
        returns = calculate_returns(episode)

        for state, action, G in returns:
            returns_sum[state] += G
            returns_count[state] += 1

    for state in returns_sum:
        values[state] = returns_sum[state] / returns_count[state]

    return values


# ============================================================
# 5. EPSILON-SOFT POLICY
# ============================================================

def epsilon_greedy_action(state, Q):
    actions = list(ACTIONS.keys())

    if random.random() < EPSILON:
        return random.choice(actions)

    q_values = [Q[(state, action)] for action in actions]
    best_value = max(q_values)

    best_actions = [
        action for action in actions
        if Q[(state, action)] == best_value
    ]

    return random.choice(best_actions)


# ============================================================
# 6. MONTE CARLO CONTROL
# ============================================================

def monte_carlo_control():
    Q = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for episode_number in range(EPISODES):

        # Generate episode using epsilon-soft policy
        episode = []
        state = START

        for _ in range(MAX_STEPS):
            action = epsilon_greedy_action(state, Q)
            next_state, reward, done = step(state, action)

            episode.append((state, action, reward))
            state = next_state

            if done:
                break

        returns = calculate_returns(episode)
        visited = set()

        # First-visit MC control update
        for state, action, G in returns:
            state_action = (state, action)

            if state_action not in visited:
                returns_sum[state_action] += G
                returns_count[state_action] += 1

                Q[state_action] = (
                    returns_sum[state_action] /
                    returns_count[state_action]
                )

                visited.add(state_action)

    return Q


# ============================================================
# 7. TEMPORAL-DIFFERENCE LEARNING: TD(0)
# ============================================================

def td_zero(episodes):
    V = defaultdict(float)

    for _ in range(episodes):
        state = START

        for _ in range(MAX_STEPS):
            action = random.choice(list(ACTIONS.keys()))

            next_state, reward, done = step(state, action)

            # TD(0) update
            if done:
                target = reward
            else:
                target = reward + GAMMA * V[next_state]

            V[state] = V[state] + ALPHA * (
                target - V[state]
            )

            state = next_state

            if done:
                break

    return V


# ============================================================
# 8. DISPLAY LEARNED POLICY
# ============================================================

def display_policy(Q):
    print("\nLEARNED MONTE CARLO POLICY")
    print("--------------------------")

    symbols = {
        "UP": "^",
        "DOWN": "v",
        "LEFT": "<",
        "RIGHT": ">"
    }

    for row in range(5):
        line = []

        for col in range(5):
            state = (row, col)

            if GRID[row][col] == "#":
                line.append("#")
            elif state == START:
                line.append("S")
            elif state == GOAL:
                line.append("G")
            else:
                action = max(
                    ACTIONS.keys(),
                    key=lambda a: Q[(state, a)]
                )
                line.append(symbols[action])

        print(" ".join(line))


# ============================================================
# 9. NUMERICAL TD(0) EXERCISE FROM ASSIGNMENT
# ============================================================

def numerical_td_example():
    print("\nTD(0) NUMERICAL EXERCISE")
    print("-----------------------")

    V_current = 5
    reward = 3
    V_next = 8
    alpha = 0.2
    gamma = 0.9

    target = reward + gamma * V_next
    td_error = target - V_current

    updated_value = V_current + alpha * td_error

    print("Current value:", V_current)
    print("Reward:", reward)
    print("Next-state value:", V_next)
    print("Learning rate:", alpha)
    print("Discount factor:", gamma)

    print("\nTarget = 3 + (0.9 * 8) =", target)
    print("TD Error =", target, "-", V_current, "=", td_error)

    print(
        "Updated value = 5 + (0.2 * 5.2) =",
        round(updated_value, 2)
    )


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TASK 3: MONTE CARLO AND TD(0) LEARNING")
    print("AUTONOMOUS WAREHOUSE DELIVERY ROBOT")
    print("=" * 60)

    # Generate sample episodes for MC prediction
    print("\nGenerating Monte Carlo episodes...")

    episodes = [
        generate_episode()
        for _ in range(500)
    ]

    # First-Visit MC
    first_values = first_visit_mc(episodes)

    # Every-Visit MC
    every_values = every_visit_mc(episodes)

    print("\nFIRST-VISIT MONTE CARLO STATE VALUES")
    for state in sorted(first_values):
        print(state, ":", round(first_values[state], 3))

    print("\nEVERY-VISIT MONTE CARLO STATE VALUES")
    for state in sorted(every_values):
        print(state, ":", round(every_values[state], 3))

    # Monte Carlo Control
    print("\nTraining Monte Carlo Control...")
    Q = monte_carlo_control()

    print("Monte Carlo Control training completed.")

    display_policy(Q)

    # TD(0)
    print("\nTraining TD(0)...")
    td_values = td_zero(EPISODES)

    print("\nTD(0) STATE VALUES")
    for state in sorted(td_values):
        print(state, ":", round(td_values[state], 3))

    # Numerical exercise
    numerical_td_example()

    # MC vs TD comparison
    print("\nMONTE CARLO VS TD(0)")
    print("-------------------")
    print("Monte Carlo:")
    print("- Learns from complete episodes.")
    print("- Uses actual returns.")
    print("- Does not bootstrap.")
    print("- Can have higher variance.")

    print("\nTD(0):")
    print("- Learns from individual transitions.")
    print("- Uses reward and estimated next-state value.")
    print("- Uses bootstrapping.")
    print("- Can update before an episode ends.")

    print("\nCONCLUSION")
    print("TD(0) is suitable for repeated warehouse navigation")
    print("because it can learn after each transition without")
    print("waiting for a complete delivery episode.")

    print("\nTASK 3 PROGRAM COMPLETED.")
