import numpy as np
from typing import Literal, Tuple
from operator import itemgetter

class Memory(object):
    """
    Class implementing the memory buffer

    """
    def __init__(self, algo_type: Literal["on-policy", "off-policy"], buffer_size: int=int(1e6), need_next_actions: bool=False, seed: int=42) -> None:
        self.algo_type = algo_type
        self.buffer_size = buffer_size
        self.need_next_actions = need_next_actions
        self.reset()

        self.seed = seed
        self.rng = np.random.default_rng(seed)


    def update(self, state, action, reward: float, next_state, done: bool, next_action=None) -> None:
        """ 
        Adds a new experience (s, a, r, s', a') to the memory. Depending on algorithm type the 
        behaviour is slightly different
             on-policy: append sample to the current episode and start a new episode 
                        if it has terminated
            off-policy: insert sample to the last position and if the buffer is full, 
                        overwrite the oldest sample 
        """
        if self.algo_type == "on-policy":
            self.states[self.pointer].append(state)
            self.actions[self.pointer].append(action)
            self.rewards[self.pointer].append(reward)
            self.next_states[self.pointer].append(next_state)
            self.dones[self.pointer].append(done)
            if self.need_next_actions:
                self.next_actions[self.pointer].append(next_action)
            if done:
                self.pointer += 1
                self.states.append([])
                self.actions.append([])
                self.rewards.append([])
                self.next_states.append([])
                self.dones.append([])
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


    def sample(self, num: int=None) -> Tuple:
        """
        Samples num experiences from the memory. If num is empty, all experiences are taken.
            num:
                    For 'on-policy' algorithms num is the number of epsiodes
                    For 'off-policy' algorithms num is the number of actual experiences
                    If no num is provided (default), all experiences are utilized
        """
        if not num:
            num = self.pointer

        indices = self.rng.choice(self.pointer, num, replace=False)
        states = list(itemgetter(*indices)(self.states))
        actions = list(itemgetter(*indices)(self.actions))
        rewards = list(itemgetter(*indices)(self.rewards))
        next_states = list(itemgetter(*indices)(self.next_states))
        dones = list(itemgetter(*indices)(self.dones))
        if self.need_next_actions:
            next_actions = itemgetter(*indices)(self.next_actions)
            return states, actions, rewards, next_states, dones, next_actions
        else:
            return states, actions, rewards, dones, next_states    

    def reset(self) -> None:
        """
        Resets the memory by discarding all existing samples
        """
        if self.algo_type == "on-policy":
            self.pointer = 0 # episode
            self.states = [[]]
            self.actions = [[]]
            self.rewards = [[]]
            self.next_states = [[]]
            self.dones = [[]]
            if self.need_next_actions:
                self.next_actions = [[]]
        else:
            self.pointer = 0 # point in buffer
            self.states = self.buffer_size*[None]
            self.actions = self.buffer_size*[None]
            self.rewards = self.buffer_size*[None]
            self.next_states = self.buffer_size*[None]
            self.dones = self.buffer_size*[None]
            if self.need_next_actions:
                self.next_actions = self.buffer_size*[None]


if __name__ == "__main__":
    # exmaple
    import torch
    import copy
    torch.manual_seed(42)

    mem_on_policy = Memory('on-policy')
    mem_off_policy = Memory('off-policy')
    for k in range(100):
        # create random experience
        state = torch.randn((3, 3))
        action = torch.randn(1)
        reward = torch.randn(1)
        next = torch.randn((3, 3))
        done = 1 if (k + 1) % 20 == 0 else 0

        # create deep-copy of experience
        state_ = copy.deepcopy(state)
        action_ = copy.deepcopy(action)
        reward_ = copy.deepcopy(reward)
        next_ = copy.deepcopy(next)
        done_ = copy.deepcopy(done)

        # append experience to memory
        mem_on_policy.update(state, action, reward, next, done)
        mem_off_policy.update(state_, action_, reward_, next_, done_)

    states, actions,rewards, next_states, dones = mem_on_policy.sample()
    states_, actions_, rewards_, next_states_, dones_ = mem_off_policy.sample()
    
    num_episodes = len(states)
    num_experiences = len(states_)

    assert(num_episodes == 5)
    assert(num_experiences == 100)

    print("All tests have passed successfully!")


        

        
