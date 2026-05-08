import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import scipy.signal

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, num_agents=1):
        self.obs_buf = np.zeros((size, num_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, num_agents), dtype=np.float32)
        self.adv_buf = np.zeros((size, num_agents), dtype=np.float32)
        self.rew_buf = np.zeros((size, num_agents), dtype=np.float32)
        self.ret_buf = np.zeros((size, num_agents), dtype=np.float32)
        self.val_buf = np.zeros((size, num_agents), dtype=np.float32)
        self.logp_buf = np.zeros((size, num_agents), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.num_agents = num_agents
        self.path_start_idx_per_agent = np.zeros(num_agents, dtype=int)

    def store(self, obs_list, act_list, rew_list, val_list, logp_list):
        for i in range(self.num_agents):
            self.obs_buf[self.ptr, i] = obs_list[i]
            self.act_buf[self.ptr, i] = act_list[i]
            self.rew_buf[self.ptr, i] = rew_list[i]
            self.val_buf[self.ptr, i] = val_list[i]
            self.logp_buf[self.ptr, i] = logp_list[i]
        self.ptr += 1

    def finish_path_agent(self, agent_idx, last_val=0.0):
        """结束单个智能体的轨迹（用于中途碰撞/到达时的独立终止）。"""
        start = self.path_start_idx_per_agent[agent_idx]
        end = self.ptr
        if start >= end:
            return
        path_slice = slice(start, end)
        rews = np.append(self.rew_buf[path_slice, agent_idx], last_val)
        vals = np.append(self.val_buf[path_slice, agent_idx], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice, agent_idx] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice, agent_idx] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx_per_agent[agent_idx] = self.ptr

    def finish_path(self, last_val_list):
        for i in range(self.num_agents):
            self.finish_path_agent(i, last_val_list[i])
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        self.path_start_idx_per_agent = np.zeros(self.num_agents, dtype=int)
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

class PPO:
    def __init__(self, policy, pi_lr=3e-4, vf_lr=3e-4, gamma=0.99, lam=0.95,
                 clip_ratio=0.2, target_kl=0.01, train_pi_iters=10, train_v_iters=10,
                 max_grad_norm=0.5, use_gpu=False, entropy_coef=0.0):
        self.policy = policy
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.max_grad_norm = max_grad_norm
        self.use_gpu = use_gpu
        self.entropy_coef = entropy_coef

        # 策略优化器（所有参数：共享backbone + 策略头）
        self.optimizer = Adam(policy.parameters(), lr=pi_lr)
        # 价值函数独立优化器（仅价值头参数，避免 v_iters 污染共享 backbone）
        # 兼容 TransformerPolicyV2 (value_head + value) 和 MLPPolicyDirect (value_net)
        if hasattr(policy, 'value_head') and hasattr(policy, 'value'):
            value_params = (list(policy.value_head.parameters()) +
                            list(policy.value.parameters()))
        elif hasattr(policy, 'value_net'):
            value_params = list(policy.value_net.parameters())
        else:
            value_params = list(policy.parameters())  # fallback: all params
        self.v_optimizer = Adam(value_params, lr=vf_lr)

    def compute_loss_v(self, obs, ret):
        _, values = self.policy.forward(obs)
        return ((values.squeeze(-1) - ret) ** 2).mean()

    def compute_loss_pi(self, obs, act, adv, logp_old):
        dist = self.policy.get_dist(obs)
        # act shape: (N,), distribution batch_shape: (N, act_dim)
        # Must unsqueeze to (N, act_dim) so log_prob broadcasts element-wise, not cross-product
        logp = dist.log_prob(act.unsqueeze(-1)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        loss_pi = loss_pi - self.entropy_coef * entropy  # entropy bonus: encourage exploration
        approx_kl = (logp_old - logp).mean().item()
        return loss_pi, approx_kl, entropy.item()

    def update(self, data):
        obs = data['obs']  # (size, num_agents, obs_dim)
        act = data['act']  # (size, num_agents)
        ret = data['ret']  # (size, num_agents)
        adv = data['adv']  # (size, num_agents)
        logp_old = data['logp']  # (size, num_agents)

        size, num_agents = obs.shape[:2]

        # 展平为 (size * num_agents, ...)
        obs_flat = obs.view(-1, obs.shape[-1])
        act_flat = act.view(-1)
        ret_flat = ret.view(-1)
        adv_flat = adv.view(-1)
        logp_old_flat = logp_old.view(-1)

        # 标准化优势
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # 检查数据是否有 NaN
        if torch.isnan(obs_flat).any() or torch.isnan(act_flat).any() or torch.isnan(ret_flat).any() or torch.isnan(adv_flat).any() or torch.isnan(logp_old_flat).any():
            print(f"⚠️  PPO 更新输入包含 NaN:")
            print(f"   obs_flat NaN: {torch.isnan(obs_flat).any()}")
            print(f"   act_flat NaN: {torch.isnan(act_flat).any()}")
            print(f"   ret_flat NaN: {torch.isnan(ret_flat).any()}")
            print(f"   adv_flat NaN: {torch.isnan(adv_flat).any()}")
            print(f"   logp_old_flat NaN: {torch.isnan(logp_old_flat).any()}")
            return {'pi_loss': 0.0, 'v_loss': 0.0, 'kl': 0.0, 'entropy': 0.0, 'pi_iters': 0}

        if self.use_gpu:
            obs_flat, act_flat, ret_flat, adv_flat, logp_old_flat = obs_flat.cuda(), act_flat.cuda(), ret_flat.cuda(), adv_flat.cuda(), logp_old_flat.cuda()

        pi_losses, kl_vals, entropy_vals = [], [], []

        # 策略更新
        for _ in range(self.train_pi_iters):
            self.optimizer.zero_grad()
            loss_pi, kl, entropy = self.compute_loss_pi(obs_flat, act_flat, adv_flat, logp_old_flat)

            # 检查损失是否有 NaN
            if torch.isnan(loss_pi).any():
                print(f"⚠️  策略损失包含 NaN: {loss_pi}")
                break

            if kl > self.target_kl:
                break
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            pi_losses.append(loss_pi.item())
            kl_vals.append(kl)
            entropy_vals.append(entropy)

        v_losses = []

        # 价值网络更新（使用独立优化器，避免污染共享 backbone）
        for _ in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            loss_v = self.compute_loss_v(obs_flat, ret_flat)

            # 检查损失是否有 NaN
            if torch.isnan(loss_v).any():
                print(f"⚠️  价值损失包含 NaN: {loss_v}")
                break

            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
            self.v_optimizer.step()
            v_losses.append(loss_v.item())

        return {
            'pi_loss': float(np.mean(pi_losses)) if pi_losses else 0.0,
            'v_loss': float(np.mean(v_losses)) if v_losses else 0.0,
            'kl': float(np.mean(kl_vals)) if kl_vals else 0.0,
            'entropy': float(np.mean(entropy_vals)) if entropy_vals else 0.0,
            'pi_iters': len(pi_losses),
        }