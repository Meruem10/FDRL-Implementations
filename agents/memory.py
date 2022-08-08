from logging import BufferingFormatter
import torch 
import numpy as np

class Memory():
    """
    Class implementing the Buffer

    TO-DO:
        -Implement a method to track the episodes in order to average over several episodes for On-Policy method
        or when using Monte Carlo approximation for the target value function
    """
    def __init__(self, buffer_size: int=1e6, need_next_actions: bool=False, seed: int=42) -> None:
        self.buffer_size = buffer_size
        self.reset()

        self.seed = seed
        np.random.seed(42)


    def sample(self, num: int) -> tuple:
        indices = np.random.randint(0, self.pointer, num)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        if self.next_states:
            next_actions = self.next_actions[indices]
            return states, actions, rewards, next_states, next_actions
        else:
            return states, actions, rewards, next_states      


    def reset(self):
        self.pointer = 0
        self.states = [0]*self.buffer_size
        self.actions = [0]*self.buffer_size
        self.rewards = [0]*self.buffer_size
        self.next_states = [0]*self.buffer_size
        self.dones = [0]*self.buffer_size
        if self.need_next_actions:
            self.next_actions = [0]*self.buffer_size


if __name__ == "__main__":
    pass
        
