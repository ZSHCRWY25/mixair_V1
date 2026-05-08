import numpy as np
from env.BaseUAV import BaseUAV

class FreeUAV(BaseUAV):
    """
    自由无人机：使用三维 RVO 算法进行避撞，动作由 RVO 控制器给出（三维速度）。
    包含独立的 RVO 计算方法。
    """
    def __init__(self, id, starting, destination, waypoints, n_points, init_acc,
                 priority=1, dt=1, vel_max=2.0*np.ones(3), goal_threshold=3, radius=1.2,
                 safety_radius=None, **kwargs):
        super().__init__(id, starting, destination, waypoints, n_points, init_acc,
                         priority, dt, vel_max, goal_threshold, radius, **kwargs)
        self.uav_type = 'free'
        self.uav_type_num = 0        # 0 表示自由无人机
        self.goal = self.destination.copy()   # 自由无人机目标点
        self.safety_radius = safety_radius if safety_radius is not None else radius * 2   # 避撞安全距离
        self.reached_goal = False

    def compute_preferred_velocity(self):
        """计算指向目标点的期望速度"""
        direction = self.goal - self.state
        dist = np.linalg.norm(direction)
        if dist < self.goal_threshold:
            self.reached_goal = True
            return np.zeros(3)
        if dist > 0:
            direction = direction / dist
        speed = min(self.max_speed, dist * 0.5)   # 减速接近
        return direction * speed

    def compute_3d_velocity_obstacle(self, other):
        """
        计算与另一自由无人机的三维速度障碍锥。
        返回 (cone_axis, cone_angle) 或 (None, None)。
        """
        rel_pos = other.state - self.state
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-5:
            return None, None

        # 相对速度
        rel_vel = self.vel - other.vel
        closing_speed = np.dot(rel_vel, rel_pos) / dist
        if closing_speed < 0:   # 正在远离
            return None, None

        combined_radius = self.safety_radius + getattr(other, 'safety_radius', other.radius * 2)
        sin_theta = combined_radius / dist
        if sin_theta >= 1:
            # 已经碰撞或非常接近
            return None, None

        cone_axis = rel_pos / dist
        cone_angle = np.arcsin(sin_theta)
        return cone_axis, cone_angle

    def compute_avoidance_velocity(self, others):
        """
        基于当前状态和其他自由无人机列表，使用三维 RVO 计算避撞速度。
        自由无人机只感知其他自由无人机，不感知航路无人机（优先级设计）。
        返回建议的速度向量。
        """
        pref_vel = self.compute_preferred_velocity()
        if self.reached_goal:
            return np.zeros(3)

        # 生成候选速度（球面采样，4×4=16个方向+2特殊=18候选，约为原来的1/3）
        num_samples = 4
        phi = np.linspace(0, np.pi, num_samples)
        theta = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        candidates = []
        for p in phi:
            for t in theta:
                x = self.max_speed * np.sin(p) * np.cos(t)
                y = self.max_speed * np.sin(p) * np.sin(t)
                z = self.max_speed * np.cos(p)
                candidates.append(np.array([x, y, z]))
        # 加入首选速度和零速度
        candidates.append(pref_vel)
        candidates.append(np.zeros(3))
        # 按与首选速度的接近程度排序（欧氏距离）
        candidates.sort(key=lambda v: np.linalg.norm(v - pref_vel))

        best_vel = candidates[0]
        min_cost = float('inf')
        for vel in candidates:
            if np.linalg.norm(vel) > self.max_speed:
                continue
            collision = False
            cost = np.linalg.norm(vel - pref_vel)
            for other in others:
                if other.id == self.id or other.reached_goal:
                    continue
                cone_axis, cone_angle = self.compute_3d_velocity_obstacle(other)
                if cone_axis is None:
                    continue
                # 检查相对速度是否在锥内
                rel_vel = vel - other.vel
                norm_rv = np.linalg.norm(rel_vel)
                if norm_rv < 1e-6:
                    continue
                angle = np.arccos(np.clip(np.dot(rel_vel, cone_axis) / norm_rv, -1.0, 1.0))
                if angle < cone_angle:
                    collision = True
                    cost += 1000   # 碰撞惩罚
                    break
            if not collision and cost < min_cost:
                min_cost = cost
                best_vel = vel
        return best_vel

    def update_with_rvo(self, other_free_uavs):
        """
        使用 RVO 计算速度并更新状态。
        :param other_free_uavs: 其他自由无人机列表（不包含自身）
        """
        vel_cmd = self.compute_avoidance_velocity(other_free_uavs)
        self.move(vel_cmd)

    # 兼容原有接口：move_forward 可以接收三维速度指令（若需要外部指定）
    def move_forward(self, act, *args, **kwargs):
        """
        若外部直接给出速度指令，则使用此方法。
        act 应为三维速度向量。
        """
        self.move(act)