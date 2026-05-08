import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPolicy(nn.Module):
    def __init__(self,
                 self_state_dim=10,
                 free_obs_dim=11,
                 max_free=10,
                 embed_dim=66,
                 nhead=4,
                 num_layers=2,
                 mlp_hidden=[128, 64],
                 action_scale=2.0,
                 device='cpu'):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.free_obs_dim = free_obs_dim
        self.max_free = max_free
        self.action_scale = action_scale
        self.device = device

        # 自由无人机嵌入层
        self.intruder_embed = nn.Linear(free_obs_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_free, embed_dim))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        combined_dim = embed_dim + self_state_dim
        layers = []
        in_dim = combined_dim
        for h in mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 可学习的标准差（对数形式）
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        """
        obs: (batch, self_state_dim + max_free * free_obs_dim)
        returns: action_mean (batch,), value (batch,)
        """
        batch_size = obs.shape[0]

        # 分离自身状态和自由无人机观测
        self_state = obs[:, :self.self_state_dim]                                    # (batch, self_state_dim)
        intruder_flat = obs[:, self.self_state_dim:]                                 # (batch, max_free * free_obs_dim)
        intruders = intruder_flat.view(batch_size, self.max_free, self.free_obs_dim) # (batch, max_free, free_obs_dim)

        # 嵌入
        intruder_feat = self.intruder_embed(intruders)                               # (batch, max_free, embed_dim)
        intruder_feat = intruder_feat + self.pos_embed[:, :self.max_free, :]

        # 生成掩码（填充位置的特征全为零）
        mask = (intruders.sum(dim=-1) != 0)                                          # (batch, max_free)
        key_padding_mask = ~mask                                                     # True 表示需要忽略

        # Transformer 编码
        encoded = self.transformer(intruder_feat, src_key_padding_mask=key_padding_mask)  # (batch, max_free, embed_dim)

        # 池化（平均有效位置）
        mask_expand = mask.unsqueeze(-1).float()                                     # (batch, max_free, 1)
        sum_encoded = (encoded * mask_expand).sum(dim=1)                             # (batch, embed_dim)
        count = mask_expand.sum(dim=1) + 1e-8                                        # (batch, 1)
        pooled = sum_encoded / count                                                 # (batch, embed_dim)

        # 拼接自身状态
        combined = torch.cat([pooled, self_state], dim=-1)                           # (batch, embed_dim + self_state_dim)

        # 输出动作均值和价值
        action_mean = torch.tanh(self.mlp(combined)) * self.action_scale             # (batch, 1)
        value = self.value_head(combined)                                            # (batch, 1)
        return action_mean.squeeze(-1), value.squeeze(-1)

    def get_dist(self, obs):
        """返回动作分布（正态分布）"""
        action_mean, _ = self.forward(obs)

        # print(f"Action Mean: {action_mean}")

        std = torch.exp(self.log_std).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, std)

    def v(self, obs):
        """只返回状态价值"""
        _, value = self.forward(obs)
        return value