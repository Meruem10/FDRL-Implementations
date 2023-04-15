import torch
import torch.optim as optim
from torch.distributions import Normal, Categorical

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Union, Tuple

from ann import ANN
from memory import Memory

# # Debugger Tutorial: https://www.youtube.com/watch?v=R3smFr6W8jI
# import debugpy
# debugpy.listen(5678)
# print("Waiting for Debugger")
# debugpy.wait_for_client()
# print("Debugger is attached")


class Reinforce(object):
    """
    Class implementing the 'REINFORCE' on-policy reinforcement learning algorithm

    TBD: For now we assume scalar actions, i.e. include multivariate actions
    """

    action: Union[int, float, torch.tensor]

    def __init__(
        self,
        policy_net: ANN,
        optimizer: torch.optim,
        gamma: float = 0.99,
        standardize_returns: bool = True,
        entropy_coef: float = 0.0,
        action_space: str = Literal["continuous", "discrete"],
        max_action: float = np.float32("inf"),
        min_action: float = -np.float32("inf"),
        sample_action: bool = True,
        seed: int = 42,
    ) -> None:
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.memory = Memory(
            "on-policy",
            custom_fields={"logprob_actions": None, "entropy": None},
        )
        self.gamma = gamma
        self.max_action = max_action
        self.min_action = min_action
        self.sample_action = sample_action
        self.seed = seed

        self.action_space = action_space
        self.dtype = self.policy_net.dtype
        self.np_dtype = (
            np.float32 if self.dtype == torch.float32 else np.float64
        )
        self.device = self.policy_net.device

        self.standardize_returns = standardize_returns
        self.eps = 1e-6
        self.entropy_coef = entropy_coef

        self._reset()

        # set training mode
        self.policy_net.train()

        torch.manual_seed(self.seed)

    def act(self, state) -> Tuple["action", "action"]:
        """returns action and its logprob sampled from the policy-network"""
        # build policy distribution
        x = torch.from_numpy(state.astype(self.np_dtype)).unsqueeze(0)
        pdparams = self.policy_net(x)
        if self.action_space == "continuous":
            pd = Normal(loc=pdparams[0], scale=pdparams[1])
        else:
            if self.policy_net.softmax_output:
                pd = Categorical(pdparams)
            else:
                pd = Categorical(logits=pdparams)

        # sample from policy distribution
        if self.action_space == "continuous" and not self.sample_action:
            # basically greedy policy
            a = pd.mean
        else:
            # substitute for eps-greedy policy
            a = pd.sample()

        # clamp reward
        a = torch.clamp(a, min=self.min_action, max=self.max_action)

        # log-normal/log-Categorical distrubion
        logprob_a = pd.log_prob(a)

        # entropy
        entropy = pd.entropy()

        if self.action_space == "discrete":
            return int(a.item()), logprob_a, entropy
        else:
            return a.item(), logprob_a, entropy

    def update(
        self,
        reward: float,
        logprob_action: torch.tensor,
        done: bool,
        entropy: torch.tensor = None,
    ) -> None:
        """Updates the memory with the required quantities of the new experience"""
        self.memory.update(
            reward=reward,
            logprob_actions=logprob_action,
            done=done,
            entropy=entropy,
        )

    def train(self) -> float:
        """Performs a training step using all sampled trajectories so far"""
        samples = self.memory.sample()

        # calculate loss for single trajectory
        num_eps = self.memory.pointer
        loss = 0.0
        for ep in range(num_eps):
            logprob_actions = samples["logprob_actions"][ep]
            rewards = samples["rewards"][ep]
            assert len(logprob_actions) == len(
                rewards
            ), "Length of quantities inside episode do not match"
            len_eps = len(rewards)

            # calculate Rt(tau)
            Rt = 0.0
            traj_returns = len_eps * [None]
            for i in range(len_eps - 1, -1, -1):
                Rt = rewards[i] + self.gamma * Rt
                traj_returns[i] = Rt

            traj_returns = torch.tensor(
                traj_returns, dtype=self.dtype, device=self.device
            )

            if self.standardize_returns:
                traj_returns = (traj_returns - traj_returns.mean()) / (
                    traj_returns.std() + self.eps
                )

            # calculate loss
            logprob_actions = torch.cat(logprob_actions)
            obj = torch.sum(traj_returns * logprob_actions)

            entropy = torch.cat(samples["entropy"][ep]).mean()
            obj += self.entropy_coef * entropy

            loss -= obj  # PyTorch minimizes cost

        # average over all trajectories
        loss = loss / num_eps

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # discard experiences from previous policy
        self._reset()

        return loss.item()

    def _reset(self) -> None:
        """Discards the data gathered from previous policy"""
        self.memory.reset()


if __name__ == "__main__":
    # example
    import gymnasium as gym

    torch.manual_seed(42)

    # choose
    gamma = 0.99
    entropy_coef = 0.0
    num_eps = 350  # 300
    maxlen_ep = 500  # 200
    render = False
    train_freq = 5  # train after 'train_freq' episodes
    sample_action = True
    seed = 42

    # define environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    in_dim = env.observation_space.shape[0]  # 4
    success_thres = env.spec.reward_threshold  # 195

    # define agent
    policy_net = ANN(
        num_inputs=in_dim,
        num_outputs=2,
        hidden_layer_dims=[64, 32],
        softmax_output=False,
        dropout=0.0,
        use_batch_norm=False,
        weight_init="xavier_normal",
        weight_init_kw={"gain": 1.0},
    )
    optimizer = optim.Adam(params=policy_net.parameters(), lr=1e-2)
    policy = Reinforce(
        policy_net=policy_net,
        optimizer=optimizer,
        gamma=gamma,
        entropy_coef=entropy_coef,
        action_space="discrete",
        sample_action=sample_action,
        seed=seed,
    )

    # main loop
    for ep in range(1, num_eps + 1):
        # reset environment
        state, _ = env.reset(seed=seed + ep)
        for t in range(maxlen_ep):
            # sample action from current policy
            a, log_a, entropy = policy.act(state)

            # apply sampled action to the environment
            state, reward, done, _, _ = env.step(a)
            if t == maxlen_ep - 1:
                done = True

            # update memory buffer
            policy.update(
                reward=reward, logprob_action=log_a, done=done, entropy=entropy
            )

            # render
            if render:
                screen = env.render()
                plt.imshow(screen)

            # has episode ended
            if done:
                break

        # train
        if ep % train_freq == 0:
            total_reward = np.mean(
                list(map(lambda x: sum(x), policy.memory.rewards[:-1]))
            )
            if total_reward > success_thres:
                print(
                    f"Congratulations total reward of {total_reward:.2f}, environment has been solved successfully!"
                )
                break
            loss = policy.train()
            print(
                f"Episode: {ep}, Total reward: {total_reward:.2f}, Training loss: {loss:.2f}"
            )
    env.close()

    print("All tests have passed successfully!")
