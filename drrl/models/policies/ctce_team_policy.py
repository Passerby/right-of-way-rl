import time

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from gymnasium import spaces

from .ctde_team_policy import layer_init, MLPGaussianActor

class CTCETeamPolicy(nn.Module):

    def __init__(self, cfg) -> None:
        super(CTCETeamPolicy, self).__init__()

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

        self.critic_z = nn.Sequential(
            layer_init(nn.Linear(n_input_channels, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )

        self.actor_z = nn.Sequential(
            layer_init(nn.Linear(n_input_channels, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )


        self.critic = nn.Sequential(
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=0.01)
        )

        self.actor = nn.Sequential(
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

    def max_pool_shared_z(self, z):
        shared_z, other_z = torch.split(z, [64, 192], dim=1)
        shared_z = shared_z.reshape(-1, self.max_agents, 64)
        shared_z = torch.max(shared_z, dim=1)[0].reshape(-1, 1, 64)
        shared_z = torch.tile(shared_z, (1, self.max_agents, 1)).reshape(-1, 64)
        return_z = torch.cat([shared_z, other_z], dim=1)
        return return_z

    def forward(self, actor_obs, critic_obs, done=None):
        batch_size = actor_obs.shape[0]
        actor_obs = actor_obs.reshape(-1, self.n_input_channels)
        critic_obs = critic_obs.reshape(-1, self.n_input_channels)

        az = self.max_pool_shared_z(self.actor_z(actor_obs))
        cz = self.max_pool_shared_z(self.critic_z(critic_obs))

        normal_dist, _ = self.actor(az)
        actions = normal_dist.sample()
        log_probs = normal_dist.log_prob(actions)
        value = self.critic(cz)

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
            az = self.max_pool_shared_z(self.actor_z(obs))
            d, _ = self.actor(az)
        return d.mean

    def increment_update(self):
        self._update_count += 1
        self._update_time.data = torch.tensor(time.time(), dtype=torch.float64)
