
import random

# Calculate reward for the agent
def calculate_reward(action, next_position, treasure, enemy, exit_point):

    reward = -1

    # TODO 1: Positive reward for collecting treasure
    if next_position == treasure:
        reward += 10

    # TODO 2: Negative reward for encountering an enemy
    if enemy is not None and next_position == enemy:
        reward -= 20

    # TODO 3: Positive reward for reaching the exit
    if next_position == exit_point:
        reward += 50

    # TODO 4: Return final reward
    return reward


# Test the assignment's given example
reward = calculate_reward(
    "RIGHT", (2, 3), (2, 3), None, (4, 4)
)

print("Game Reward:", reward)


# Maze environment
class MazeGame:

    def __init__(self, walls, treasures, enemy, moving_enemy=False):
        self.start = (0, 0)
        self.exit = (4, 4)
        self.walls = set(walls)
        self.treasures = set(treasures)
        self.enemy = enemy
        self.moving_enemy = moving_enemy
        self.position = self.start
        self.steps = 0
        self.collected = 0
        self.collisions = 0
        self.done = False

    def move(self, action):

        row, col = self.position

        movements = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }

        dr, dc = movements[action]
        next_position = (row + dr, col + dc)

        # Check maze boundaries and walls
        if (not (0 <= next_position[0] < 5 and
                 0 <= next_position[1] < 5)
                or next_position in self.walls):

            self.steps += 1
            return -10

        self.position = next_position
        self.steps += 1

        reward = -1

        # Collect treasure
        if self.position in self.treasures:
            self.treasures.remove(self.position)
            self.collected += 1
            reward += 10

        # Enemy collision
        if self.position == self.enemy:
            reward -= 20
            self.collisions += 1
            self.done = True
            return reward

        # Reach exit
        if self.position == self.exit:
            reward += 50
            self.done = True

        # Move enemy in scenario 2
        if self.moving_enemy and not self.done:
            self.enemy = (1, 2) if self.enemy == (1, 3) else (1, 3)

        return reward


def run_scenario(name, walls, treasures, enemy, moving_enemy=False):

    print("\n", name)

    game = MazeGame(
        walls, treasures, enemy, moving_enemy
    )

    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    total_reward = 0

    random.seed(42)

    while not game.done and game.steps < 30:

        action = random.choice(actions)
        reward = game.move(action)
        total_reward += reward

    success = game.position == game.exit

    print("Steps:", game.steps)
    print("Treasures collected:", game.collected)
    print("Collisions:", game.collisions)
    print("Game completed:", success)
    print("Total reward:", total_reward)


# Scenario 1: Normal maze
run_scenario(
    "Scenario 1 - Normal Maze",
    walls={(0, 2), (2, 1), (3, 3)},
    treasures={(1, 1), (3, 2)},
    enemy=(1, 3)
)

# Scenario 2: Moving enemy
run_scenario(
    "Scenario 2 - Moving Enemy",
    walls={(0, 2), (2, 1), (3, 3)},
    treasures={(1, 1), (3, 2)},
    enemy=(1, 3),
    moving_enemy=True
)

# Scenario 3: Dynamic obstacle
run_scenario(
    "Scenario 3 - Dynamic Obstacles",
    walls={(0, 2), (2, 1), (3, 3), (2, 3)},
    treasures={(1, 1), (3, 2)},
    enemy=(1, 3)
)
