import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class TransformerPolicyV2(nn.Module):
    """
    创新的多智能体策略网络 - 使用 Transformer 处理可变长度的自由 UAV
    
    观测构成：
    - 自身状态：10维
    - 最威胁航路UAV：11维（10维状态 + 1维TTC）
    - 自由UAV集合（可变长度，最多5个）：每个9维（8维特征 + 1维TTC）
    
    处理流程：
    1. 编码自身状态 + 最威胁航路UAV
    2. 使用 Transformer 处理可变长度的自由UAV
    3. 拼接所有包含池化后的自由UAV特征
    4. 通过 MLP 输出动作和价值
    
    这是该项目的创新点！
    """
    
    def __init__(self,
                 self_state_dim=10,
                 other_route_dim=16,  # 8rel + ttc + prog + type_indicator + heading_align + 4 conflict_type onehot
                 free_uav_dim=10,     # 8rel + ttc + path_crossing_time
                 max_free_uavs=5,
                 embed_dim=64,
                 nhead=4,
                 num_layers=2,
                 mlp_hidden=[256, 128],
                 device='cpu'):
        super().__init__()
        
        self.self_state_dim = self_state_dim
        self.other_route_dim = other_route_dim
        self.free_uav_dim = free_uav_dim
        self.max_free_uavs = max_free_uavs
        self.embed_dim = embed_dim
        self.device = device
        
        # 【改进】只对自由 UAV 进行嵌入和 Transformer 处理
        # 航路 UAV 观测保持原始状态
        
        # 1. 嵌入层：仅用于自由 UAV（使用 Xavier 初始化）
        self.free_uav_embed = nn.Linear(free_uav_dim, embed_dim)
        nn.init.xavier_uniform_(self.free_uav_embed.weight)
        nn.init.zeros_(self.free_uav_embed.bias)
        
        # LayerNorm 来稳定嵌入输出
        self.embed_norm = nn.LayerNorm(embed_dim)
        
        # 2. Transformer 编码器处理可变长度的自由 UAV
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
            dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 位置编码（用于 Transformer）
        self.register_buffer(
            'pos_embed',
            self._get_pos_encoding(max_free_uavs, embed_dim)
        )
        
        # 4. 【改进】新的组合维度
        # 航路 UAV 观测: self_state_dim + other_route_dim = 10 + 16 = 26
        # 自由 UAV 池化特征: embed_dim = 64
        route_uav_dim = self_state_dim + other_route_dim  # 26
        combined_dim = route_uav_dim + embed_dim  # 22 + 64 = 86
        
        # 5. MLP 头部（输入为路由 UAV 观测 + 池化的自由 UAV 特征）
        layers = []
        in_dim = combined_dim
        for h in mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        
        # 策略头：输出动作均值
        self.policy_head = nn.Sequential(*layers)
        self.action_mean = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Tanh()  # 限制输出到 [-1, 1] 范围
        )
        
        # 价值头：输出状态价值
        layers_v = []
        in_dim = combined_dim
        for h in mlp_hidden:
            layers_v.append(nn.Linear(in_dim, h))
            layers_v.append(nn.ReLU())
            in_dim = h
        self.value_head = nn.Sequential(*layers_v)
        self.value = nn.Linear(in_dim, 1)
        
        # 可学习的标准差
        self.log_std = nn.Parameter(torch.zeros(1))
        
        # 初始化所有权重
        self.apply(self._init_weights)
        
        self.to(device)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _get_pos_encoding(self, seq_len, d_model):
        """生成位置编码"""
        pos = np.arange(seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        return torch.from_numpy(pe).float()
    
    def forward(self, obs):
        """
        前向传播 - 【改进】航路 UAV 观测保持原始状态，仅自由 UAV 经 Transformer

        Args:
            obs: 观测向量，形状 (batch, 10 + 16 + 5*10)
                 = (batch, 10 + 16 + 50) = (batch, 76)
                 其中前 10 是自身状态，次 16 是最威胁航路UAV（含4维冲突类型one-hot），
                 后 50 是最多5个自由UAV的观测

        Returns:
            action_mean: (batch, 1)
            value: (batch, 1)
        """
        # 检查输入是否有 NaN
        if torch.isnan(obs).any():
            print(f"⚠️  输入观测包含 NaN: {obs}")
            obs = torch.nan_to_num(obs, nan=0.0)
        
        batch_size = obs.shape[0]
        
        # 【改进】分解观测
        # 航路 UAV 观测保持原始状态
        route_uav_obs = obs[:, :self.self_state_dim + self.other_route_dim]  # (batch, 26)
        free_uavs_flat = obs[:, self.self_state_dim + self.other_route_dim:]  # (batch, 50)
        
        # 重塑自由UAV观测为 (batch, max_free_uavs, free_uav_dim)
        free_uavs = free_uavs_flat.view(batch_size, self.max_free_uavs, self.free_uav_dim)
        
        # 1. 仅嵌入和处理自由 UAV
        free_uavs_emb = self.embed_norm(self.free_uav_embed(free_uavs))  # (batch, max_free_uavs, embed_dim)
        
        # 【修复】检查嵌入输出是否有 NaN 或 INF
        if torch.isnan(free_uavs_emb).any() or torch.isinf(free_uavs_emb).any():
            print(f"⚠️  free_uavs_emb 包含 NaN/Inf，使用 nan_to_num 修复")
            free_uavs_emb = torch.clamp(torch.nan_to_num(free_uavs_emb, nan=0.0, posinf=1.0, neginf=-1.0), -10, 10)
        
        # 2. 添加位置编码
        free_uavs_emb = free_uavs_emb + self.pos_embed[:self.max_free_uavs].unsqueeze(0)
        
        # 3. 生成掩码（识别有效的自由UAV观测）
        mask = (free_uavs.abs().sum(dim=-1) > 1e-6)  # (batch, max_free_uavs)
        src_key_padding_mask = ~mask                 # True 表示要忽略
        
        # 4. Transformer 编码 - 处理全 padding 序列
        encoded_free_uavs = self.transformer(
            free_uavs_emb,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch, max_free_uavs, embed_dim)
        
        # 【修复】检查 Transformer 输出并修复 NaN
        if torch.isnan(encoded_free_uavs).any() or torch.isinf(encoded_free_uavs).any():
            print(f"⚠️  Transformer 输出包含 NaN/Inf")
            encoded_free_uavs = torch.clamp(
                torch.nan_to_num(encoded_free_uavs, nan=0.0, posinf=1.0, neginf=-1.0),
                -10, 10
            )
        
        # 5. 池化有效的自由UAV特征（均值池化）
        mask_expand = mask.unsqueeze(-1).float()  # (batch, max_free_uavs, 1)
        sum_encoded = (encoded_free_uavs * mask_expand).sum(dim=1)  # (batch, embed_dim)
        count = mask_expand.sum(dim=1)                             # (batch, 1)
        
        # 【修复】数值稳定的池化
        valid_count = torch.clamp(count, min=1.0)  # 至少为1，避免除零
        pooled_free_uavs = sum_encoded / valid_count  # (batch, embed_dim)
        
        # 当没有有效UAV时，强制设置为零
        no_valid_mask = (count.squeeze(-1) < 0.5)  # (batch,)
        pooled_free_uavs = torch.where(
            no_valid_mask.unsqueeze(-1),
            torch.zeros_like(pooled_free_uavs),
            pooled_free_uavs
        )
        
        # 【修复】最后检查池化输出
        if torch.isnan(pooled_free_uavs).any() or torch.isinf(pooled_free_uavs).any():
            print(f"⚠️  pooled_free_uavs 包含 NaN/Inf，修复为零")
            pooled_free_uavs = torch.zeros_like(pooled_free_uavs)
        
        # 6. 【改进】拼接：航路 UAV 原始观测 + 池化的自由 UAV 特征
        combined = torch.cat([
            route_uav_obs,        # (batch, 21) - 航路 UAV 原始观测
            pooled_free_uavs      # (batch, embed_dim) - 池化的自由 UAV 特征
        ], dim=-1)  # (batch, 21 + embed_dim)
        
        # 【修复】检查拼接结果
        if torch.isnan(combined).any() or torch.isinf(combined).any():
            print(f"⚠️  combined 包含 NaN/Inf，使用 clamp")
            combined = torch.clamp(torch.nan_to_num(combined, nan=0.0), -10, 10)
        
        # 7. MLP 前向传播
        policy_feat = self.policy_head(combined)
        action_mean = 2.0 * self.action_mean(policy_feat)  # Scale Tanh[-1,1] to [-2,2] to match action space
        
        value_feat = self.value_head(combined)
        value = self.value(value_feat)
        
        # 【修复】检查输出并修复 NaN
        if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
            print(f"⚠️  action_mean 包含 NaN/Inf，修复")
            action_mean = torch.clamp(torch.nan_to_num(action_mean, nan=0.0), -10, 10)
        
        if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"⚠️  value 包含 NaN/Inf，修复")
            value = torch.clamp(torch.nan_to_num(value, nan=0.0), -10, 10)
        
        return action_mean, value
    
    def get_dist(self, obs):
        """返回动作分布"""
        action_mean, _ = self.forward(obs)
        
        # 【修复】确保 std 在合理范围内（避免过小或过大）
        std = torch.exp(torch.clamp(self.log_std, -2, 2))  # std 在 [0.135, 7.39] 范围
        std = torch.clamp(std, min=1e-4, max=2.0)  # 进一步限制标准差范围
        
        # 确保 action_mean 在合理范围
        action_mean = torch.clamp(action_mean, -10, 10)
        
        dist = Normal(action_mean, std)
        return dist
    
    def get_action(self, obs):
        """采样动作"""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        _, value = self.forward(obs)
        
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)


# =======================
# 对比版本：保留原始的直接 MLP（方便对比实验）
# =======================

class MLPPolicyDirect(nn.Module):
    """
    直接版本：所有观测直接输入 MLP
    这个版本用于对比实验，验证 Transformer 的效果
    """
    
    def __init__(self, obs_dim=76, act_dim=1, hidden_sizes=[256, 128], device='cpu'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # 策略网络
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        layers.append(nn.Tanh())  # 限制输出到 [-1, 1]，forward 中乘 2 扩展到 [-2, 2]
        self.mean_net = nn.Sequential(*layers)

        # 价值网络
        layers_v = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers_v.append(nn.Linear(in_dim, h))
            layers_v.append(nn.ReLU())
            in_dim = h
        layers_v.append(nn.Linear(in_dim, 1))
        self.value_net = nn.Sequential(*layers_v)

        # 可学习的标准差
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.to(device)

    def forward(self, obs):
        """直接通过 MLP"""
        action_mean = 2.0 * self.mean_net(obs)  # Scale Tanh[-1,1] to [-2,2]
        value = self.value_net(obs)
        return action_mean, value

    def get_dist(self, obs):
        """返回动作分布"""
        action_mean, _ = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        return dist

    def get_action(self, obs):
        """采样动作"""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        dist = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        _, value = self.forward(obs)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)
