
import numpy as np

np.random.seed(7)

true_rewards = [1.0, 1.5, 2.0, 1.2]
n_arms = len(true_rewards)
epsilon = 0.1
steps = 1000

counts = np.zeros(n_arms)
estimates = np.zeros(n_arms)
total_reward = 0.0

for step in range(steps):

    # Explore or exploit
    if np.random.rand() < epsilon:
        action = np.random.randint(n_arms)
    else:
        action = int(np.argmax(estimates))

    # Generate simulated reward
    reward = np.random.normal(true_rewards[action], 1.0)

    # Update selected arm count
    counts[action] += 1

    # Update estimated average reward
    estimates[action] += (
        reward - estimates[action]
    ) / counts[action]

    # Update total reward
    total_reward += reward

print("Arm selections:", counts.astype(int))
print("Estimated rewards:", np.round(estimates, 3))
print("Average reward:", round(total_reward / steps, 3))
