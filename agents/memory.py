import numpy as np
from typing import Literal, Tuple, Dict
from operator import itemgetter

# # Debugger Tutorial: https://www.youtube.com/watch?v=R3smFr6W8jI
# import debugpy
# debugpy.listen(5678)
# print("Waiting for Debugger")
# debugpy.wait_for_client()
# print("Debugger is attached")

class Memory(object):
    """
    Class implementing the memory buffer

    To-Do: 
        - [DONE] Add options for general fields in memory, e.g. log_probs required for REINFORCE
        - Rename fields to non-plural
        - Add option not declare the actual fields needed to save, e.g. REINFORCE does not require to save (s, a, s')
    """
    def __init__(self, algo_type: Literal["on-policy", "off-policy"], buffer_size: int=int(1e6), need_next_actions: bool=False, seed: int=42, custom_fields: Dict={}) -> None:
        self.algo_type = algo_type
        self.buffer_size = buffer_size
        self.need_next_actions = need_next_actions
        self.custom_fields = custom_fields
        self.reset()

        self.seed = seed
        self.rng = np.random.default_rng(seed)


    def update(self, state=None, action=None, reward: float=None, next_state=None, done: bool=None, next_action=None, **kwargs) -> None:
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
            for field, data in kwargs.items():
                if field in self.custom_fields:
                    getattr(self, field)[self.pointer].append(data)
            if done:
                self.pointer += 1
                self.states.append([])
                self.actions.append([])
                self.rewards.append([])
                self.next_states.append([])
                self.dones.append([])
                if self.need_next_actions:
                    self.next_actions.append([])
                for field, data in kwargs.items():
                    getattr(self, field).append([])
        else:
            self.states[self.pointer] = state
            self.actions[self.pointer] = action
            self.rewards[self.pointer] = reward
            self.next_states[self.pointer] = next_state
            self.dones[self.pointer] = done
            if self.need_next_actions:
                self.next_actions[self.pointer] = next_action
            for field, data in kwargs.items():
                if field in self.custom_fields:
                    getattr(self, field)[self.pointer] = data
 
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

        samples = {}
        indices = self.rng.choice(self.pointer, num, replace=False)
        samples["states"] = list(itemgetter(*indices)(self.states))
        samples["actions"] = list(itemgetter(*indices)(self.actions))
        samples["rewards"] = list(itemgetter(*indices)(self.rewards))
        samples["next_states"] = list(itemgetter(*indices)(self.next_states))
        samples["dones"] = list(itemgetter(*indices)(self.dones))
        if self.need_next_actions:
            samples["next_actions"] = list(itemgetter(*indices)(self.next_actions))
        for field in self.custom_fields:
            samples[field] = list(itemgetter(*indices)(getattr(self, field)))

        # single episode consistency with API [[episode]]
        if len(indices) == 1:
            for key in samples.keys():
                samples[key] = [samples[key]]

        return samples

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
            for field in self.custom_fields.keys():
                setattr(self, field, [[]])
        else:
            self.pointer = 0 # point in buffer
            self.states = self.buffer_size*[None]
            self.actions = self.buffer_size*[None]
            self.rewards = self.buffer_size*[None]
            self.next_states = self.buffer_size*[None]
            self.dones = self.buffer_size*[None]
            if self.need_next_actions:
                self.next_actions = self.buffer_size*[None]
            for field in self.custom_fields.keys():
                setattr(self, field, self.buffer_size*[None])


if __name__ == "__main__":
    # exmaple
    import torch
    import copy
    torch.manual_seed(42)

    custom_fields = {"log_probs": None, "entropy": None}
    mem_on_policy = Memory('on-policy', custom_fields=custom_fields)
    mem_off_policy = Memory('off-policy', custom_fields=custom_fields)
    for k in range(100):
        # create random experience
        state = torch.randn((3, 3))
        action = torch.randn(1)
        reward = torch.randn(1)
        next = torch.randn((3, 3))
        done = 1 if (k + 1) % 20 == 0 else 0
        for field in custom_fields.keys():
            custom_fields[field] = torch.randn(1)

        # create deep-copy of experience
        state_ = copy.deepcopy(state)
        action_ = copy.deepcopy(action)
        reward_ = copy.deepcopy(reward)
        next_ = copy.deepcopy(next)
        done_ = copy.deepcopy(done)
        custom_fields_ = copy.deepcopy(custom_fields)

        # append experience to memory
        mem_on_policy.update(state, action, reward, next, done, **custom_fields)
        mem_off_policy.update(state_, action_, reward_, next_, done_, **custom_fields_)

    samples = mem_on_policy.sample()
    samples_ = mem_off_policy.sample()
    
    num_episodes = len(samples["states"])
    num_experiences = len(samples_["states"])

    assert(num_episodes == 5)
    assert(num_experiences == 100)

    assert('log_probs' in samples.keys())
    assert('log_probs' in samples_.keys())
    assert('entropy' in samples.keys())
    assert('entropy' in samples_.keys())

    print("All tests have passed successfully!")


        

        
