import numpy as np
from env.BaseUAV import BaseUAV

class RouteUAV(BaseUAV):
    """
    航路无人机：只能沿当前航向加速/减速，动作输入为一维加速度标量。
    """
    def __init__(self, id, starting, destination, waypoints, n_points, init_acc,
                 priority=1, dt=1, vel_max=2.0*np.ones(3), goal_threshold=3, radius=1.2,
                 min_speed=0.5, **kwargs):
        super().__init__(id, starting, destination, waypoints, n_points, init_acc,
                         priority, dt, vel_max, goal_threshold, radius, **kwargs)
        self.min_speed = min_speed   # 允许的最小速度（防止停转）
        self.uav_type = 'route'
        self.uav_type_num = 1        # 1 表示航路无人机

    def apply_acceleration(self, acc_scalar):
        """
        应用一维加速度更新状态。
        :param acc_scalar: 加速度值，范围 [-a_max, a_max]
        """
        # 计算当前航向单位向量（从 previous_des 指向 current_des）
        direction = self.current_des - self.previous_des
        norm_dir = np.linalg.norm(direction)
        if norm_dir < 1e-6:
            # 若航段长度为零，则保持原速度方向（或期望方向）
            if np.linalg.norm(self.vel) > 1e-6:
                heading = self.vel / np.linalg.norm(self.vel)
            else:
                heading = self.angles_to_direction(self.relative(self.state, self.current_des)[1])
        else:
            heading = direction / norm_dir

        # 当前速度大小
        current_speed = np.linalg.norm(self.vel)

        # 应用加速度更新速度大小
        new_speed = current_speed + acc_scalar * self.dt
        new_speed = np.clip(new_speed, self.min_speed, self.max_speed)

        # 合成新速度向量
        new_vel = new_speed * heading

        # 调用基类的 move 更新位置
        self.move(new_vel)

    # 兼容原有接口：move_forward 接收标量加速度
    def move_forward(self, act, *args, **kwargs):
        self.apply_acceleration(act)