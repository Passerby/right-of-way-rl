import time

import numpy as np
import torch
import torch.nn as nn


class PPO:

    def __init__(self, cfg, policy_network):
        self.cfg = cfg
        self.policy = policy_network
        self.player_num = cfg['policy']['placeholder']['max_agents']

        self.epoch = self.cfg.epoch
        self.clip_range = self.cfg.clip_range
        self.clip_range_vf = self.cfg.clip_range_vf
        self.vf_coef = self.cfg.vf_coef
        self.ent_coef = self.cfg.ent_coef
        self.actor_coef = self.cfg.actor_coef
        self.learning_rate = self.cfg.learning_rate
        self.max_grad_norm = self.cfg.max_grad_norm

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)

    @property
    def update_count(self):
        if not hasattr(self.policy, 'update_count'):
            return self.policy.module.update_count
        return self.policy.update_count

    def increment_update(self):
        if not hasattr(self.policy, 'increment_update'):
            self.policy.module.increment_update()
        else:
            self.policy.increment_update()

    def compute_loss(self, states: torch.Tensor, critic_states: torch.Tensor, old_actions: torch.Tensor, old_logprobs: torch.Tensor,
                     old_values: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor, done: torch.Tensor = None):
        mini_batch_sise = states.shape[0]
        # Compute new log probabilities and state values
        _new_actions, _new_sample_log_probs, new_values, probs = self.policy(states, critic_states)
        # new_log_probs = self.policy.action_distribution.log_prob(old_actions)

        # old_actions shape is (batch_size, numplayer * action shape) reshape to (batch_size * numplayer,  action shape)
        old_actions = old_actions.reshape(mini_batch_sise * self.player_num, -1)
        old_logprobs = old_logprobs.reshape(mini_batch_sise * self.player_num, -1)
        returns = returns.reshape(mini_batch_sise * self.player_num, -1)
        old_values = old_values.reshape(mini_batch_sise * self.player_num, -1)
        not_done = (1 - done).reshape(mini_batch_sise * self.player_num, -1)

        new_logprobs = probs.log_prob(old_actions)

        # Calculate the ratio
        logratio = (new_logprobs - old_logprobs).mean(axis=-1) # (batch_size * numplayer,  1)
        ratio = logratio.exp()

        clipfracs = []
        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > self.clip_range).float().mean().item()]

        advantages = advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate surrogate losses
        surr1 = -advantages * ratio
        surr2 = -advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        # Policy gradient loss
        policy_loss = torch.max(surr1, surr2)

        # policy is not real excute when done
        policy_loss_with_done = policy_loss # * not_done

        policy_loss_with_done = policy_loss_with_done.mean()
        policy_loss_with_done = policy_loss_with_done * self.actor_coef

        # Value function loss
        v_loss_unclipped = (new_values - returns)**2
        v_clipped = old_values + torch.clamp(new_values - old_values, -self.clip_range_vf, self.clip_range_vf)
        v_loss_clipped = (v_clipped - returns)**2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss_max_with_done = v_loss_max # * not_done

        vf_coef = self.vf_coef * min((self.update_count + 35.0) / 70.0, 0.85)
        v_loss_with_done = vf_coef * v_loss_max_with_done.mean()

        # entropy
        # entropy = self.policy.action_distribution.entropy().mean()
        entropy = probs.entropy()
        # entropy_with_done = (entropy * not_done).mean()
        entropy_with_done = entropy.mean()
        entropy_loss_with_done = self.ent_coef * entropy_with_done

        # Total loss
        loss = policy_loss_with_done - entropy_loss_with_done + v_loss_with_done

        # backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        total_grads = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return [
            loss.item(),
            policy_loss_with_done.item(),
            v_loss_with_done.item(),
            entropy_loss_with_done.item(),
            entropy.mean().item(),
            total_grads.item(),
            clipfracs[0],
            old_approx_kl.item(),
            approx_kl.item(),
        ]

    def learn(self, dataset):

        loss_names = [
            "loss",
            "pg_loss",
            "vf_loss",
            "entropy_loss",
            "entropy",
            "a_global_grads",
            "clipfrac",
            'old_approxkl',
            "approxkl",
        ]
        mb_loss_vals = []

        sgd_time = []

        # advantages = dataset['adv']
        # advantages = advantages.reshape(-1)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for i in range(self.epoch):
            sgd_t = time.time()

            batch_size = dataset['actor_obs'].shape[0]
            minibatch_size = batch_size // 4
            b_inds = np.arange(batch_size)
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                losses = self.compute_loss(
                    states=dataset['actor_obs'][mb_inds],
                    critic_states=dataset['critic_obs'][mb_inds],
                    # myself_vector=dataset['myself_vector'],
                    old_actions=dataset['action'][mb_inds],
                    # old_pre_actions=dataset['pre_action'],
                    old_logprobs=dataset['logprob'][mb_inds],
                    old_values=dataset['value'][mb_inds],
                    returns=dataset['returns'][mb_inds],
                    advantages=dataset['adv'][mb_inds],
                    done=dataset['done'][mb_inds])
            mb_loss_vals.append(losses)
            sgd_time.append(time.time() - sgd_t)

        model_log_dict = {}
        loss_vals = np.mean(mb_loss_vals, axis=0)
        for (loss_val, loss_name) in zip(loss_vals, loss_names):
            model_log_dict[loss_name] = loss_val

        train_log_dict = {"sgd_time": np.mean(sgd_time)}

        return model_log_dict, train_log_dict
