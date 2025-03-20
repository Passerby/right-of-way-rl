import time

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from gymnasium import spaces


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):

    def _distribution(self, x):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, x, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(x)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):

    def __init__(self, input_dim, act_dim):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = nn.Linear(input_dim, act_dim)

    def _distribution(self, x):
        mu = self.mu_net(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class CTDETeamPolicy(nn.Module):

    def __init__(self, cfg) -> None:
        super(CTDETeamPolicy, self).__init__()

        self.cfg = cfg['policy']
        self.placeholder = cfg['policy']['placeholder']

        # basic env information
        self.max_agents = self.placeholder['max_agents']
        # self.max_teams = self.placeholder['max_teams']

        # input information
        self.observation_space = self.placeholder['observation_space']
        self.action_space = self.placeholder['action_space']

        print(self.observation_space, self.action_space)

        self._update_count = nn.Parameter(torch.tensor(0), requires_grad=False)
        self._update_time = nn.Parameter(torch.tensor(time.time(), dtype=torch.float64), requires_grad=False)

        n_input_channels = self.observation_space[0]
        self.n_input_channels = n_input_channels

        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_input_channels, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=0.01)
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_input_channels, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            MLPGaussianActor(256, self.action_space)
        )


    @property
    def update_count(self):
        return self._update_count.item()

    @property
    def update_time(self):
        return self._update_time.item()

    def forward(self, actor_obs, critic_obs, done=None):
        batch_size = actor_obs.shape[0]
        actor_obs = actor_obs.reshape(-1, self.n_input_channels)
        critic_obs = critic_obs.reshape(-1, self.n_input_channels)

        normal_dist, _ = self.actor(actor_obs)
        actions = normal_dist.sample()
        log_probs = normal_dist.log_prob(actions)
        value = self.critic(critic_obs)

        if done is not None:
            not_done = (1 - done).reshape(-1, 1)
            actions *= not_done
            log_probs *= not_done
            value *= not_done

        return actions, log_probs, value, normal_dist

    def batch_step(self, dataset_dict):
        with torch.no_grad():
            batch_size = dataset_dict['actor_obs'].shape[0] # batch_size, n player * obs shape

            actions, log_probs, value, probs = self.forward(dataset_dict['actor_obs'], dataset_dict['critic_obs'], dataset_dict['done'])
            results = (actions, log_probs, value)

            results = torch.cat(list(results), dim=-1)

            # Insert update_count to the last column
            update_count = self._update_count
            update_count = update_count * torch.ones((batch_size * self.max_agents, 1), dtype=torch.float32, device=update_count.device)
            results = torch.cat([results, update_count], dim=-1)

            # batch size, n player, output shape
            results = results.reshape(batch_size, self.max_agents, -1)

        return results

    def action(self, obs):
        with torch.no_grad():
            d, _ = self.actor(obs)
        return d.mean

    def increment_update(self):
        self._update_count += 1
        self._update_time.data = torch.tensor(time.time(), dtype=torch.float64)
