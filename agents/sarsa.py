import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Union, Dict, Tuple

from ann import ANN
from memory import Memory

# # Debugger Tutorial: https://www.youtube.com/watch?v=R3smFr6W8jI
# import debugpy
# debugpy.listen(5678)
# print("Waiting for Debugger")
# debugpy.wait_for_client()
# print("Debugger is attached")

class SARSA(object):
    """
    Class implementing the 'SARSA' on-policy reinforcement learning algorithm

    TBD: For now we assume scalar actions, i.e. include multivariate actions 
    """
    action: Union[int, float, torch.tensor]

    def __init__(self, q_value_net: ANN, optimizer: torch.optim, gamma: float=0.99, target_strategy: str=Literal["TD-Backup", "Monte-Carlo"], standardize_returns: bool=True, epsilon: float=0.9, seed: int=42) -> None:
        self.q_value_net = q_value_net
        self.optimizer = optimizer
        self.memory = Memory('on-policy', custom_fields={"next_actions": None}) 
        self.gamma = gamma
        self.target_strategy = target_strategy
        
        self.epsilon = epsilon
        self.n_actions = self.q_value_net.num_outputs
        self.seed = seed
        self.dtype = self.q_value_net.dtype
        self.np_dtype = np.float32 if self.dtype == torch.float32 else np.float64
        self.device = self.q_value_net.device

        self.standardize_returns = standardize_returns

        self._reset()

        # set training mode
        self.value_net.train()

        torch.manual_seed(self.seed)
        

    def act(self, state) -> 'action':
        """ Utilizes epsilon-greedy strategy to select action from the q_value_net"""
        x = torch.from_numpy(state.astype(self.np_dtype)).unsqueeze(0).to(self.device)
        if torch.rand(1).item() < self.epsilon:
            a = np.random.choice(self.n_actions, 1).item()
        else:
            self.q_value_net.eval()
            with torch.no_grad():
                q_vals = self.q_value_net(x)
                a = torch.argmax(q_vals).item()
            self.q_value_net.train()
        
        return a

    def update(self, state, action: action, reward: float, next_state, next_action: action) -> None:
        """ Updates the memory with the required quantities of the new experience """
        self.memory.update(state=state, action=action, reward=reward, next_state=next_state, next_action=next_action)

    def train(self) -> float:
        pass

    
    def _calculate_targets(self) -> Dict[list[list]]:
        """ Calculates the targets using either: TD-Backup or Monte-Carlo """
        samples = self.memory.sample()
        num_eps = self.memory.pointer

        if self.target_strategy == "Monte-Carlo":
            # discard last episode, if it has not finished yet (all previous episodes should have finished)
            # cannot calculate an MC estimate of unfinished trajectories
            if not samples["dones"][-1][-1]:
                num_eps -= 1
                for key in samples.keys():
                    samples[key].pop()
            
            # check if there is at least one remaining episode
            if not samples["dones"]:
                raise NotImplementedError
            
            # calculate MC-targets using the actual received cumulative rewards 
            # in the trajectory
            samples["targets"] = []
            for ep in range(num_eps):
                rewards = samples["rewards"][ep]
                targets = np.cumsum(rewards[::-1])

                samples["targets"].append(targets)            
        else:
            """ TO-DO: Vectorize the calculation, i.e. instead of sequence processing, make simultaneously"""
            # TD-Backup
            samples["targets"] = []
            for ep in range(num_eps):
                len_eps = len(samples["rewards"][ep])

                targets = len_eps*[None]
                for i in range(len_eps):
                    x_next = torch.from_numpy(samples["next_states"][ep][i].astype(self.np_dtype)).unsqueeze(0).to(self.device)
                    a_next = samples["actions"][ep][i]
                    with torch.no_grad():
                        q_vals_next = self.q_value_net(x_next)
                        q_val_next = q_vals_next[a_next]
                    
                    targets[i] = samples["rewards"][ep][i] + (1 - samples["dones"][ep][i])*self.gamma*q_val_next

                samples["targets"].append(targets)            
        
        return samples
    
    def _update_epsilon(self):
        """ Update the epsilon as training goes on (i.e. decreased)"""
        pass

    def _reset(self) -> None:
        """ Discards the data gathered from previous policy """
        self.memory.reset()
    

if __name__ == "__main__":

    print("All tests have passed successfully!")