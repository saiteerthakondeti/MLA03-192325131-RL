
gamma = 0.8

# Two possible next states and their transition probabilities
probabilities = [0.6, 0.4]
rewards = [5, -2]
next_state_values = [10, 4]


def bellman_return(probabilities, rewards, next_state_values, gamma):
    total = 0.0

    for i in range(len(probabilities)):
        # Calculate probability-weighted reward
        weighted_reward = probabilities[i] * (
            rewards[i] + gamma * next_state_values[i]
        )

        # Add contribution to total
        total += weighted_reward

    return total


print("Bellman return:", bellman_return(
    probabilities, rewards, next_state_values, gamma
))
