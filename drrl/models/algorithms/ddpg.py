import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class BatchRewardWhitening:
    def __init__(self, alpha=0.99):
        self.alpha = alpha  # 滑动平均的衰减因子
        self.mean = 0
        self.var = 1.0 # 初始化为1以避免除以零的错误

    def update_stats(self, rewards):
        # rewards 的形状应为 [batch_size, agent_num, 1]
        batch_mean = rewards.mean().item()  # 计算每个 agent 的均值
        batch_var = rewards.var().item()  # 计算每个 agent 的方差

        # 更新均值和方差使用滑动平均
        self.mean = self.alpha * self.mean + (1 - self.alpha) * batch_mean
        self.var = self.alpha * self.var + (1 - self.alpha) * batch_var

    def whiten_rewards(self, rewards):
        # 白化奖励
        std = torch.sqrt(torch.tensor(self.var) + 1e-8)  # 防止除以零
        whitened_rewards = (rewards - self.mean) / std
        return whitened_rewards

class DDPG:

    def __init__(self, cfg, policy_network):
        self.cfg = cfg
        self.policy = policy_network.module
        self.player_num = cfg['policy']['placeholder']['max_agents']

        self.batch_size = self.cfg.batch_size
        self.epoch = self.cfg.epoch
        self.learning_rate = self.cfg.learning_rate
        self.q_lr = self.cfg.q_lr
        self.max_grad_norm = self.cfg.max_grad_norm
        # self.alpha = 0.2
        self.tau = 0.005
        self.gamma = 0.99

        self.q_optimizer = torch.optim.Adam(list(self.policy.qf1.parameters()), lr=self.q_lr, eps=1e-5)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.learning_rate, eps=1e-5)

        self.reward_processor = BatchRewardWhitening()

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

    @property
    def alpha(self):
        if not hasattr(self.policy, 'alpha'):
            return self.policy.module.alpha
        return self.policy.alpha

    def compute_loss(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor):
        # critic loss
        with torch.no_grad():
            next_actions, next_state_log_pi, _ = self.policy.actor.get_action(next_obs)
            qf1_next_target = self.policy.qf1_target(torch.cat((next_obs, next_actions), dim=-1))
            self.reward_processor.update_stats(rewards)
            whitened_rewards = self.reward_processor.whiten_rewards(rewards)
            next_q_value = (whitened_rewards + (1 - dones) * self.gamma * (qf1_next_target)).view(-1)

        qf1_a_values = self.policy.qf1(torch.cat((obs, actions), dim=-1)).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

        # optimize the model
        self.q_optimizer.zero_grad()
        qf1_loss.backward()
        qf_grads = nn.utils.clip_grad_norm_(self.policy.qf1.parameters(), self.max_grad_norm)
        self.q_optimizer.step()

        # actor loss
        actor_loss = 0
        min_qf_pi = 0
        log_pi = 0
        actor_grads = 0
        if self.policy.update_count % 8 == 0:
            for _ in range(8):
                current_actions, log_pi, _ = self.policy.actor.get_action(obs)

                actor_loss = -self.policy.qf1(torch.cat((obs, current_actions), dim=-1)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grads = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update the target network
                for param, target_param in zip(self.policy.actor.parameters(), self.policy.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.policy.qf1.parameters(), self.policy.qf1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                actor_loss = actor_loss.item()
                min_qf_pi = log_pi.mean().item()
                actor_grads = actor_grads.item()

        losses = [
            qf1_loss.item(),
            actor_loss,
            qf1_next_target.mean().item(),
            next_q_value.mean().item(),
            actor_grads,
            qf_grads.item()
        ]

        return losses

    def learn(self, dataset):
        loss_names = ["qf_loss", "actor_loss", "qf_next_target", "next_q_v", "actor_grads", "qf_grads"]
        mb_loss_vals = []
        sgd_time = []

        obs = dataset["actor_obs"].reshape(self.batch_size, self.player_num, -1)
        next_obs = dataset["next_obs"].reshape(self.batch_size, self.player_num, -1)
        actions = dataset["action"].reshape(self.batch_size, self.player_num, -1)
        rewards = dataset["reward"].reshape(self.batch_size, self.player_num, -1)
        dones = dataset["done"].reshape(self.batch_size, self.player_num, -1)

        batch_size = dataset['actor_obs'].shape[0]
        minibatch_size = batch_size // 8
        b_inds = np.arange(batch_size)
        np.random.shuffle(b_inds)
        rank = dist.get_rank()
        for start in range(0, batch_size, minibatch_size):
            sgd_t = time.time()
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            obs = torch.tensor(dataset["actor_obs"][mb_inds], dtype=torch.float32).reshape(minibatch_size, self.player_num, -1).to(rank)
            next_obs = torch.tensor(dataset["next_obs"][mb_inds], dtype=torch.float32).reshape(minibatch_size, self.player_num, -1).to(rank)
            actions = torch.tensor(dataset["action"][mb_inds], dtype=torch.float32).reshape(minibatch_size, self.player_num, -1).to(rank)
            rewards = torch.tensor(dataset["reward"][mb_inds], dtype=torch.float32).reshape(minibatch_size, self.player_num, -1).to(rank)
            dones = torch.tensor(dataset["done"][mb_inds], dtype=torch.float32).reshape(minibatch_size, self.player_num, -1).to(rank)
            loss = self.compute_loss(obs=obs, next_obs=next_obs, actions=actions, rewards=rewards, dones=dones)

            mb_loss_vals.append(loss)
            sgd_time.append(time.time() - sgd_t)

        model_log_dict = {}
        loss_vals = np.mean(mb_loss_vals, axis=0)
        for (loss_val, loss_name) in zip(loss_vals, loss_names):
            model_log_dict[loss_name] = loss_val

        # boardcast params
        self.broadcast_param()
        train_log_dict = {"sgd_time": np.mean(sgd_time)}

        return model_log_dict, train_log_dict

    def broadcast_param(self):
        # 确保你已经初始化了分布式环境
        if dist.is_initialized():
            # 从 rank 0 广播到所有进程
            dist.broadcast(self.policy._update_count, src=0)
            # dist.broadcast(self.policy._update_time, src=0)
