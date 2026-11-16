import numpy as np

main_weights = np.array([0.8, 0.6, 0.4])
target_weights = np.array([0.2, 0.3, 0.1])

tau = 0.1

# TODO 1: Contribution from main network
main_contribution = tau * main_weights

# TODO 2: Contribution from target network
target_contribution = (1 - tau) * target_weights

# TODO 3: Updated target network weights
target_weights = main_contribution + target_contribution

print("Updated Target Weights:", target_weights)