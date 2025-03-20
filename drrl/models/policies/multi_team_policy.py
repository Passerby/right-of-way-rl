import time

import torch
import torch.nn as nn
import numpy as np

from gymnasium import spaces

from drrl.models.distributions import CategoricalDistribution
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MultiTeamPolicy(nn.Module):

    def __init__(self, cfg) -> None:
        super(MultiTeamPolicy, self).__init__()

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
        # same as stable-baselines3 NatureCNN
        global_input_net = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = global_input_net(torch.as_tensor(spaces.Box(low=0, high=255, shape=self.observation_space).sample()[None]).float()).shape[1]

        self.inputs = nn.ModuleDict({
            'global_input_net': global_input_net,
            'linear': nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())
        })

        # output information
        # self.action_distribution = CategoricalDistribution(self.action_space)
        # self.proba_distribution_net = layer_init(self.action_distribution.proba_distribution_net(
        #     latent_dim=512), std=0.01)

        self.actor = layer_init(nn.Linear(512, self.action_space), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=0.01)

    @property
    def update_count(self):
        return self._update_count.item()

    @property
    def update_time(self):
        return self._update_time.item()

    def forward(self, obs):
        batch_size = obs.shape[0]
        global_input = self.inputs.global_input_net(obs / 255.0)
        hidden = self.inputs.linear(global_input)

        # logits = self.proba_distribution_net(mlp_output)

        value = self.critic(hidden)
        probs = Categorical(logits=self.actor(hidden))
        # self.action_distribution.proba_distribution(action_logits=logits)

        actions = probs.sample() # self.action_distribution.sample().reshape(batch_size, -1)
        log_probs = probs.log_prob(actions) # self.action_distribution.log_prob(actions).reshape(batch_size, -1)

        return actions.reshape(batch_size, -1), log_probs.reshape(batch_size, -1), value, probs


    def batch_step(self, dataset_dict):
        with torch.no_grad():
            batch_size = dataset_dict['obs'].shape[0]

            actions, log_probs, value, probs = self.forward(dataset_dict['obs'])
            results = (actions, log_probs, value)

            results = torch.cat(list(results), dim=-1)

            # Insert update_count to the last column
            update_count = self._update_count
            update_count = update_count * torch.ones((batch_size, 1), dtype=torch.float32, device=update_count.device)
            results = torch.cat([results, update_count], dim=-1)

            # print(results.shape)
            # print("log_prob", log_probs)

        return results

    def increment_update(self):
        self._update_count += 1
        self._update_time.data = torch.tensor(time.time(), dtype=torch.float64)
