
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# 1. INITIAL SETTINGS
# --------------------------------

gamma = 0.9
alpha = 0.01

actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# 25 states and 4 actions
policy = np.zeros((25, 4))

start = (0, 0)
treasure = (2, 2)
enemy = (1, 2)
exit_point = (4, 4)

walls = [(1, 1), (3, 3)]


# --------------------------------
# 2. STOCHASTIC POLICY
# --------------------------------

def softmax(values):
    exp_values = np.exp(values - np.max(values))
    return exp_values / np.sum(exp_values)


def choose_action(state):
    probabilities = softmax(policy[state])
    action = np.random.choice(4, p=probabilities)
    return action


# --------------------------------
# 3. DISPLAY MAZE
# --------------------------------

def display_maze():

    for row in range(5):
        for col in range(5):

            position = (row, col)

            if position == start:
                print("A", end=" ")
            elif position == treasure:
                print("T", end=" ")
            elif position == enemy:
                print("E", end=" ")
            elif position == exit_point:
                print("X", end=" ")
            elif position in walls:
                print("#", end=" ")
            else:
                print(".", end=" ")

        print()


# --------------------------------
# 4. DISCOUNTED RETURNS
# --------------------------------

def calculate_returns(rewards, gamma):

    returns = []
    G = 0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    return returns


# --------------------------------
# 5. GAME EPISODE
# --------------------------------

def play_episode():

    position = start

    states = []
    selected_actions = []
    rewards = []

    total_reward = 0
    collected = False
    completed = False

    for step in range(40):

        state = position[0] * 5 + position[1]

        action = choose_action(state)
        action_name = actions[action]

        row, col = position

        if action_name == "UP":
            row -= 1
        elif action_name == "DOWN":
            row += 1
        elif action_name == "LEFT":
            col -= 1
        elif action_name == "RIGHT":
            col += 1

        next_position = (row, col)

        reward = -1
        done = False

        # Check boundaries and walls
        if (row < 0 or row >= 5 or col < 0 or col >= 5
                or next_position in walls):

            reward = -5

        else:

            position = next_position

            # Enemy collision
            if position == enemy:
                reward = -10
                done = True

            # Treasure collection
            elif position == treasure and not collected:
                reward = 10
                collected = True

            # Exit reached
            elif position == exit_point:
                reward = 20
                completed = True
                done = True

        states.append(state)
        selected_actions.append(action)
        rewards.append(reward)

        total_reward += reward

        if done:
            break

    return states, selected_actions, rewards, total_reward, completed


# --------------------------------
# 6. REINFORCE POLICY UPDATE
# --------------------------------

def reinforce_update(states, selected_actions, rewards):

    returns = calculate_returns(rewards, gamma)

    for state, action, G in zip(
            states, selected_actions, returns):

        probabilities = softmax(policy[state])

        gradient = -probabilities
        gradient[action] += 1

        policy[state] += alpha * G * gradient


# --------------------------------
# 7. TRAIN FOR 100 EPISODES
# --------------------------------

episode_rewards = []
completed_games = 0

print("Initial Maze:")
display_maze()

print("\nTraining Started...\n")

for episode in range(100):

    states, selected_actions, rewards, total, completed = play_episode()

    reinforce_update(states, selected_actions, rewards)

    episode_rewards.append(total)

    if completed:
        completed_games += 1

    print(
        "Episode:", episode + 1,
        "| Reward:", total,
        "| Completed:", completed
    )


# --------------------------------
# 8. FINAL RESULTS
# --------------------------------

completion_rate = (completed_games / 100) * 100

print("\nTraining Completed!")
print("Total Episodes: 100")
print("Completed Games:", completed_games)
print("Completion Rate:", completion_rate, "%")
print("Average Reward:", np.mean(episode_rewards))


# --------------------------------
# 9. REWARD GRAPH
# --------------------------------

plt.plot(episode_rewards)

plt.xlabel("Training Episodes")
plt.ylabel("Episode Reward")
plt.title("REINFORCE Training Performance")

plt.grid(True)
plt.show()