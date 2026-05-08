import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim=66, act_dim=1, hidden_sizes=[256, 128], activation=nn.ReLU, device='cpu'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # 策略网络
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.mean_net = nn.Sequential(*layers)

        # 价值网络
        layers_v = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers_v.append(nn.Linear(in_dim, h))
            layers_v.append(activation())
            in_dim = h
        layers_v.append(nn.Linear(in_dim, 1))
        self.value_net = nn.Sequential(*layers_v)

        # 可学习的标准差
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.to(device)

    def forward(self, obs):
        """
        obs: (batch, obs_dim)
        returns: action_mean (batch, act_dim), value (batch, 1)
        """
        action_mean = self.mean_net(obs)
        value = self.value_net(obs)
        return action_mean, value

    def get_dist(self, obs):
        """
        返回动作分布
        """
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        return dist

    def get_action(self, obs):
        """
        采样动作
        obs: (obs_dim,) or (batch, obs_dim)
        returns: action, log_prob, value
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        _, value = self.forward(obs)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)
