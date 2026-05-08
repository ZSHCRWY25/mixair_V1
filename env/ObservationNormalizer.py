"""
观测归一化模块

实现观测值的标准化处理，提高神经网络的训练稳定性
"""

import numpy as np


class ObservationNormalizer:
    """
    观测值临时正调整器
    
    根据观测值的真实范围进行标准化处理。
    支持两种归一化方式：
    1. 固定范围归一化（基于已知的物理约束）
    2. 运行均值/标准差归一化（自适应）
    """
    
    def __init__(self, obs_dim=76, method='fixed'):
        """
        初始化归一化器

        Args:
            obs_dim: 观测维度（通常为76）
            method: 归一化方法 ('fixed' 或 'running_mean_std')
        """
        self.obs_dim = obs_dim
        self.method = method
        
        # 固定范围归一化的映射
        self.fixed_bounds = self._init_fixed_bounds()
        
        # 运行均值和标准差
        self.running_mean = np.zeros(obs_dim, dtype=np.float32)
        self.running_var = np.ones(obs_dim, dtype=np.float32)
        self.count = 0
        self.epsilon = 1e-8
    
    def _init_fixed_bounds(self):
        """
        初始化固定范围边界（相对坐标版）

        观测构成 (76维，全部为相对/自我中心表示):
        自身状态 (10维):
          [0]   current_speed        [0, 2.0]
          [1]   desired_speed        [0, 2.0]
          [2]   dist_to_waypoint     [0, 200]
          [3]   progress_ratio       [0, 1]
          [4-6] heading_dir          [-1, 1]  (单位向量)
          [7-9] desired_heading_dir  [-1, 1]
        最威胁航路UAV (16维):
          [0]   rel_pos_fwd          [-200, 200]
          [1]   rel_pos_lat          [-200, 200]
          [2]   rel_pos_alt          [-20, 20]
          [3]   dist_3d              [0, 200]
          [4]   other_speed          [0, 2.0]
          [5]   rel_vel_fwd          [-4, 4]
          [6]   rel_vel_lat          [-4, 4]
          [7]   rel_vel_alt          [-4, 4]
          [8]   TTC                  [0, 100]
          [9]   other_progress_ratio [0, 1]
          [10]  type_indicator       route=0, free=1 (always 0)
          [11]  heading_align        [-1, 1]  +1=同向, -1=正面对冲
          [12-15] conflict_type_onehot [0, 1]  4维独热: 追尾/对冲/侧交叉/斜向
        自由UAV × 5 (10维/个):
          [0]   rel_pos_fwd          [-200, 200]
          [1]   rel_pos_lat          [-200, 200]
          [2]   rel_pos_alt          [-20, 20]
          [3]   dist_3d              [0, 200]
          [4]   free_speed           [0, 2.0]
          [5]   rel_vel_fwd          [-4, 4]
          [6]   rel_vel_lat          [-4, 4]
          [7]   rel_vel_alt          [-4, 4]
          [8]   TTC                  [0, 100]
          [9]   path_crossing_time   [0, 100]
        """
        bounds = []

        # 自身状态 (10维)
        bounds.append((0.0,   2.0))   # speed
        bounds.append((0.0,   2.0))   # desired_speed
        bounds.append((0.0, 200.0))   # dist_to_wp
        bounds.append((0.0,   1.0))   # progress_ratio
        for _ in range(3):            # heading_dir (3)
            bounds.append((-1.0, 1.0))
        for _ in range(3):            # desired_heading_dir (3)
            bounds.append((-1.0, 1.0))

        # 最威胁航路UAV (16维)
        bounds.append((-200.0, 200.0))  # rel_pos_fwd
        bounds.append((-200.0, 200.0))  # rel_pos_lat
        bounds.append((-20.0,   20.0))  # rel_pos_alt
        bounds.append((0.0,    200.0))  # dist_3d
        bounds.append((0.0,      2.0))  # other_speed
        bounds.append((-4.0,     4.0))  # rel_vel_fwd
        bounds.append((-4.0,     4.0))  # rel_vel_lat
        bounds.append((-4.0,     4.0))  # rel_vel_alt
        bounds.append((0.0,    100.0))  # TTC
        bounds.append((0.0,      1.0))  # other_progress_ratio
        bounds.append((0.0,      0.0))  # type_indicator: route=0, free=1 (always 0 here)
        bounds.append((-1.0,     1.0))  # heading_align: +1=同向, -1=正面对冲
        # [新增] 冲突类型 4 维 one-hot
        bounds.append((0.0,      1.0))  # conflict class 0: 同向追尾
        bounds.append((0.0,      1.0))  # conflict class 1: 正面对冲
        bounds.append((0.0,      1.0))  # conflict class 2: 侧向交叉
        bounds.append((0.0,      1.0))  # conflict class 3: 斜向交叉

        # 自由UAV × 5 (10维/个)
        for _ in range(5):
            bounds.append((-200.0, 200.0))  # rel_pos_fwd
            bounds.append((-200.0, 200.0))  # rel_pos_lat
            bounds.append((-20.0,   20.0))  # rel_pos_alt
            bounds.append((0.0,    200.0))  # dist_3d
            bounds.append((0.0,      2.0))  # free_speed
            bounds.append((-4.0,     4.0))  # rel_vel_fwd
            bounds.append((-4.0,     4.0))  # rel_vel_lat
            bounds.append((-4.0,     4.0))  # rel_vel_alt
            bounds.append((0.0,    100.0))  # TTC
            bounds.append((0.0,    100.0))  # path_crossing_time

        return np.array(bounds, dtype=np.float32)
    
    def normalize(self, obs, method=None):
        """
        对观测进行归一化
        
        Args:
            obs: 原始观测 (shape: [..., obs_dim])
            method: 归一化方法 (默认使用初始化时指定的方法)
        
        Returns:
            归一化后的观测 (shape: 同输入)
        """
        if method is None:
            method = self.method
        
        # 确保obs是numpy array
        obs = np.asarray(obs, dtype=np.float32)
        original_shape = obs.shape
        
        # 如果是1D（单个观测），添加batch维度
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        
        if method == 'fixed':
            normalized = self._normalize_fixed(obs)
        elif method == 'running_mean_std':
            normalized = self._normalize_running_mean_std(obs)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # 恢复原始形状
        if squeeze:
            normalized = normalized.squeeze(0)
        
        return normalized
    
    def _normalize_fixed(self, obs):
        """
        使用固定范围归一化
        
        将每个维度映射到 [-1, 1] 或 [0, 1]
        """
        obs = np.asarray(obs, dtype=np.float32)
        normalized = np.zeros_like(obs)
        
        for i in range(self.obs_dim):
            lower, upper = self.fixed_bounds[i]
            center = (lower + upper) / 2.0
            scale = (upper - lower) / 2.0
            
            if scale > 0:
                # 映射到 [-1, 1]
                normalized[:, i] = (obs[:, i] - center) / scale
                # 限制到 [-1.5, 1.5] 以允许超出范围的值，但进行缩放
                normalized[:, i] = np.clip(normalized[:, i], -1.5, 1.5)
            else:
                normalized[:, i] = obs[:, i]
        
        return normalized
    
    def _normalize_running_mean_std(self, obs):
        """
        使用运行均值和标准差进行归一化
        """
        obs = np.asarray(obs, dtype=np.float32)
        batch_size = obs.shape[0]
        
        # 更新运行统计
        batch_mean = np.mean(obs, axis=0)
        batch_var = np.var(obs, axis=0)
        
        # 对标准差进行移动平均（exponential moving average）
        update_ratio = batch_size / (self.count + batch_size)
        self.running_mean = (1 - update_ratio) * self.running_mean + update_ratio * batch_mean
        self.running_var = (1 - update_ratio) * self.running_var + update_ratio * batch_var
        self.count += batch_size
        
        # 归一化
        normalized = (obs - self.running_mean) / (np.sqrt(self.running_var) + self.epsilon)
        
        # 限制到合理范围
        normalized = np.clip(normalized, -5.0, 5.0)
        
        return normalized
    
    def reset_running_stats(self):
        """重置运行统计"""
        self.running_mean = np.zeros(self.obs_dim, dtype=np.float32)
        self.running_var = np.ones(self.obs_dim, dtype=np.float32)
        self.count = 0


# 全局单例
_normalizer = None


def get_normalizer(obs_dim=66, method='fixed'):
    """获取或创建归一化器实例"""
    global _normalizer
    if _normalizer is None:
        _normalizer = ObservationNormalizer(obs_dim, method)
    return _normalizer


def normalize_obs(obs, method='fixed'):
    """
    便捷函数：对观测进行归一化
    
    Args:
        obs: 原始观测
        method: 归一化方法
    
    Returns:
        归一化后的观测
    """
    normalizer = get_normalizer(method=method)
    return normalizer.normalize(obs, method=method)
