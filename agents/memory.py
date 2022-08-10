import re
import torch 
import numpy as np
from typing import Literal
from operator import itemgetter

class Memory():
    """
    Class implementing the Buffer

    TO-DO:
        -First, make always two cases to have a minimal implementation. Afterwards make an abstract-memory class and implement
        a child class for on-policy, off-policy and Monte-Carlo methods each.
        -Implement unittests for reset, update, sample methods
    """
    def __init__(self, algo_type: Literal["on-policy", "off-policy"], buffer_size: int=1e6, need_next_actions: bool=False, seed: int=42) -> None:
        self.algo_type = algo_type
        self.buffer_size = buffer_size
        self.need_next_actions = need_next_actions
        self.reset()

        self.seed = seed
        np.random.seed(42)

    def update(self, state, action, reward: int, next_state, done: bool, next_action="nop") -> None:
        if self.algo_type == "on-policy":
            self.states[self.pointer].append(state)
            self.actions[self.pointer].append(action)
            self.rewards[self.pointer].append(reward)
            self.next_states[self.pointer].append(next_state)
            if self.need_next_actions:
                self.next_actions[self.pointer].append(next_action)
            if done:
                self.pointer += 1
                self.states.append([])
                self.actions.append([])
                self.rewards.append([])
                self.next_states.append([])
                if self.need_next_actions:
                    self.next_actions.append([])
        else:
            self.states[self.pointer] = state
            self.actions[self.pointer] = action
            self.rewards[self.pointer] = reward
            self.next_states[self.pointer] = next_state
            self.dones[self.pointer] = done
            if self.need_next_actions:
                self.next_actions[self.pointer] = next_action
        
            self.pointer = (self.pointer + 1) % self.buffer_size


    def sample(self, num: int) -> tuple:
        if self.algo_type == "on-policy":
            if self.next_actions:
                return states, actions, rewards, next_states, next_actions
            else:
                return states, actions, rewards, next_states
        else:
            indices = np.random.randint(0, self.pointer, num)
            states = list(itemgetter(*indices)(self.states))
            actions = list(itemgetter(*indices)(self.actions))
            rewards = list(itemgetter(*indices)(self.rewards))
            next_states = list(itemgetter(*indices)(self.next_states))
            dones = list(itemgetter(*indices)(self.dones))
            if self.next_states:
                next_actions = itemgetter(*indices)(self.next_actions)
                return states, actions, rewards, next_states, dones, next_actions
            else:
                return states, actions, rewards, dones, next_states    


    def reset(self):
        if self.algo_type == "on-policy":
            self.pointer = 0 # episode
            self.states = [[]]
            self.actions = [[]]
            self.rewards = [[]]
            self.next_states = [[]]
            if self.need_next_actions:
                self.next_actions = [[]]
        else:
            self.pointer = 0 # point in buffer
            self.states = [0]*self.buffer_size
            self.actions = [0]*self.buffer_size
            self.rewards = [0]*self.buffer_size
            self.next_states = [0]*self.buffer_size
            self.dones = [0]*self.buffer_size
            if self.need_next_actions:
                self.next_actions = [0]*self.buffer_size


if __name__ == "__main__":
    pass
        
