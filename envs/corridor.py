from typing import Tuple
import numpy as np

from environment import Environment


class Corridor(Environment):
    """
    Class for corridor example similar to chapter 3: SARSA.

    The state corresponding to the first index is a terminal state with reward 0.
    The state corresponding to the last index is a terminal state with reward 1.

    TO-DO:
        -Implement unittests for the individual functions
    """

    def __init__(
        self, num_tiles: int, random_init: bool = False, seed: int = 42
    ) -> None:
        if num_tiles < 3:
            raise ValueError("Cooridor must have at least 3 sections.")
        self.num_tiles = num_tiles
        self.states = [i for i in range(num_tiles)]
        self.terminal_states = [
            1 if i % (num_tiles - 1) == 0 else 0 for i in range(num_tiles)
        ]  # First and last states are terminal
        self.rewards = [0 for i in range(num_tiles)]
        self.rewards[-1] = 1  # last state has reward

        self.random_init = random_init
        self.seed = seed

    def step(self, action: int) -> Tuple[int, int, bool]:
        if action > 0:
            self.state += 1
        elif action < 0:
            self.state -= 1
        else:
            pass

        done = self.terminal_states[self.state]
        reward = self.rewards[self.state]
        return self.state, reward, done

    def reset(self) -> int:
        if self.random_init:
            np.random.seed(self.seed)
            self.state = np.random.randint(1, self.num_tiles - 1)
            self.seed += 1  # Next episode will have a different inital conidtion, but the entire experiment is still reproducible
        else:
            self.state = self.num_tiles // 2

        return self.state

    def render(self) -> None:
        output = "xxxx" * (self.num_tiles) + "x\n"
        output += "x   " * (self.num_tiles) + "x\n"
        for k in range(self.num_tiles):
            if k == self.state:
                output += "x o "
            else:
                output += "x   "
        output += "x\n"
        output += "x   " * (self.num_tiles) + "x\n"
        output += "xxxx" * (self.num_tiles) + "x\n"
        print(output)


if __name__ == "__main__":
    # Example
    np.random.seed(42)

    env = Corridor(10, random_init=True)
    states = []
    actions = []
    rewards = []
    dones = []
    states.append(env.reset())
    for k in range(100):
        action = np.random.choice([-1, 1])
        state, reward, done = env.step(action)

        env.render()

        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        states.append(state)

        if done:
            print(f"Episode has finished with reward {reward}.\n")
            env.reset()

    print("Success")
