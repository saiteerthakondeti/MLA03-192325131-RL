
import random

# --------------------------------
# MONTE CARLO PREDICTION
# --------------------------------

episodes = [
    [
        (0, -1),
        (1, -1),
        (2, 10)
    ],
    [
        (0, -1),
        (1, -1),
        (2, 10)
    ],
    [
        (0, -1),
        (1, 10)
    ]
]

gamma = 0.9

returns = {0: [], 1: [], 2: []}

# Calculate returns for each episode
for episode in episodes:
    G = 0

    for state, reward in reversed(episode):
        G = reward + gamma * G
        returns[state].append(G)

# Calculate state values
V = {}

for state in returns:
    if returns[state]:
        V[state] = sum(returns[state]) / len(returns[state])
    else:
        V[state] = 0

print("MONTE CARLO STATE VALUES")

for state, value in V.items():
    print("State", state, "Value:", round(value, 2))


# --------------------------------
# FIRST-VISIT MONTE CARLO
# --------------------------------

first_visit_returns = {0: [], 1: [], 2: []}

for episode in episodes:
    G = 0
    episode_returns = []

    for state, reward in reversed(episode):
        G = reward + gamma * G
        episode_returns.append((state, G))

    episode_returns.reverse()
    visited = set()

    for state, G in episode_returns:
        if state not in visited:
            first_visit_returns[state].append(G)
            visited.add(state)

print("\nFIRST-VISIT MONTE CARLO")

for state, values in first_visit_returns.items():
    if values:
        print("State", state, "Value:", round(sum(values) / len(values), 2))


# --------------------------------
# TD(0) NUMERICAL EXERCISE
# --------------------------------

V_current = 5
reward = 3
V_next = 8
alpha = 0.2
gamma = 0.9

V_updated = V_current + alpha * (
    reward + gamma * V_next - V_current
)

print("\nTD(0) CALCULATION")
print("Updated State Value:", round(V_updated, 2))


# --------------------------------
# TD(0) SIMPLE IMPLEMENTATION
# --------------------------------

states = [0, 1, 2]
V_td = {state: 0.0 for state in states}

transitions = [
    (0, -1, 1),
    (1, -1, 2),
    (2, 10, None)
]

for episode in range(100):
    for state, reward, next_state in transitions:
        if next_state is None:
            target = reward
        else:
            target = reward + gamma * V_td[next_state]

        V_td[state] += alpha * (target - V_td[state])

print("\nTD(0) LEARNED STATE VALUES")

for state, value in V_td.items():
    print("State", state, "Value:", round(value, 2))
