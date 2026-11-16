
import numpy as np

# Average simulated waiting time for each timing option
mean_waits = [30, 24, 18, 27]

steps = 1000
epsilon = 0.1


def run_experiment(method):
    np.random.seed(7)

    counts = np.zeros(4, dtype=int)
    estimates = np.zeros(4)
    total_reward = 0
    total_wait = 0

    for step in range(steps):

        if method == "Fixed-time":
            action = 0

        elif method == "Greedy Bandit":
            action = int(np.argmax(estimates))

        elif method == "Epsilon-greedy":
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = int(np.argmax(estimates))

        # Simulate waiting time
        waiting_time = max(
            0, np.random.normal(mean_waits[action], 3)
        )

        # Lower waiting time gives higher reward
        reward = -waiting_time

        counts[action] += 1
        total_wait += waiting_time
        total_reward += reward

        # Update reward estimate
        estimates[action] += (
            reward - estimates[action]
        ) / counts[action]

    print("\nMethod:", method)
    print("Average reward:", round(total_reward / steps, 2))
    print("Average waiting time:", round(total_wait / steps, 2))
    print("Action selections:", counts)


run_experiment("Fixed-time")
run_experiment("Greedy Bandit")
run_experiment("Epsilon-greedy")
