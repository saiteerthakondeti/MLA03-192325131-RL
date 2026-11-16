
import numpy as np

# Warehouse configuration
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
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

gamma = 0.9
theta = 0.001

states = [
    (r, c)
    for r in range(ROWS)
    for c in range(COLS)
    if (r, c) not in obstacles
]

V = {s: 0.0 for s in states}


# Environment transition model
def move(state, action):
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


# Value Iteration
while True:
    delta = 0

    for s in states:
        if s == goal:
            continue

        old_value = V[s]
        action_values = []

        for action in actions:
            next_state, reward = move(s, action)
            value = reward + gamma * V[next_state]
            action_values.append(value)

        V[s] = max(action_values)
        delta = max(delta, abs(old_value - V[s]))

    if delta < theta:
        break


# Derive the optimal policy
policy = {}

for s in states:
    if s == goal:
        policy[s] = "GOAL"
        continue

    values = {}

    for action in actions:
        next_state, reward = move(s, action)
        values[action] = reward + gamma * V[next_state]

    policy[s] = max(values, key=values.get)


# Display warehouse grid
print("WAREHOUSE GRID")
for r in range(ROWS):
    row = []
    for c in range(COLS):
        s = (r, c)

        if s == start:
            row.append(" S ")
        elif s == goal:
            row.append(" G ")
        elif s in obstacles:
            row.append(" # ")
        else:
            row.append(f"{policy[s][:1]:^3}")

    print(" ".join(row))


# Display optimal path
current = start
path = [current]

for _ in range(ROWS * COLS):
    if current == goal:
        break

    action = policy[current]
    next_state, reward = move(current, action)

    if next_state == current:
        break

    current = next_state
    path.append(current)

print("\nOptimal Path:")
print(path)

print("\nState Values:")
for s in states:
    print(s, ":", round(V[s], 2))
