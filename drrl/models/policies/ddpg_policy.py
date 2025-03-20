import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .ctde_team_policy import layer_init


LOG_STD_MAX = 10
LOG_STD_MIN = -10
class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_dim).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(action_dim))
        self.fc_logstd = nn.Linear(256, np.prod(action_dim))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((1.0 - -1.0) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((1.0 + -1.0) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class DDPGPolicy(nn.Module):

    def __init__(self, cfg) -> None:
        super(DDPGPolicy, self).__init__()

        self.cfg = cfg['policy']
        self.gamma = cfg['gamma']
        self.placeholder = cfg['policy']['placeholder']

        # basic env information
        self.max_agents = self.placeholder['max_agents']
        # input information
        self.observation_space = self.placeholder['observation_space']
        self.action_space = self.placeholder['action_space']

        self._update_count = nn.Parameter(torch.tensor(0), requires_grad=False)
        self._update_time = nn.Parameter(torch.tensor(time.time(), dtype=torch.float64), requires_grad=False)

        n_input_channels = self.observation_space[0]
        self.n_input_channels = n_input_channels

        self.qf1 = nn.Sequential(
            layer_init(nn.Linear(n_input_channels + self.action_space, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=0.01)
        )

        self.qf1_target = nn.Sequential(
            layer_init(nn.Linear(n_input_channels + self.action_space, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=0.01)
        )
        self.actor = Actor(observation_dim=[n_input_channels], action_dim=self.action_space)
        self.actor_target = Actor(observation_dim=[n_input_channels], action_dim=self.action_space)

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())


    @property
    def update_count(self):
        return self._update_count.item()

    @property
    def update_time(self):
        return self._update_time.item()


    def forward(self, x):
        pass


    def batch_step(self, dataset_dict):
        with torch.no_grad():
            batch_size = dataset_dict['actor_obs'].shape[0] # batch_size, n player * obs shape
            obs = dataset_dict['actor_obs'].reshape(-1, self.n_input_channels)

            actions, log_prob, mean = self.actor.get_action(obs)
            # actions = d.sample()

            results = actions

            # Insert update_count to the last column
            update_count = self._update_count
            update_count = update_count * torch.ones((batch_size * self.max_agents, 1), dtype=torch.float32, device=update_count.device)
            results = torch.cat([results, update_count], dim=-1)

            # batch size, n player, output shape
            results = results.reshape(batch_size, self.max_agents, -1)

        return results

    def action(self, obs):
        with torch.no_grad():
            actions, _, mean = self.actor.get_action(obs)
        return actions

    def increment_update(self):
        self._update_count += 1
        self._update_time.data = torch.tensor(time.time(), dtype=torch.float64)
