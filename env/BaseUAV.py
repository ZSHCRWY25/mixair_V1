import numpy as np
from collections import namedtuple

class BaseUAV:
    """
    无人机基类，包含所有无人机共有的属性和方法。
    """
    def __init__(self, id, starting, destination, waypoints, n_points, init_acc,
                 priority=1, dt=1, vel_max=2.5*np.ones(3), goal_threshold=3, radius=1.2, **kwargs):
        self.id = int(id)
        self.starting = np.array(starting, dtype=float)
        self.destination = np.array(destination, dtype=float)
        self.state = self.starting.copy()
        self.vel = np.zeros(3, dtype=float)
        self.vel_max = np.array(vel_max, dtype=float)   # 三维最大速度
        self.max_speed = float(np.max(self.vel_max))    # 标量最大速度（取各轴最大值，避免 norm([2.5,2.5,2.5])≈4.33 的错误）
        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]   # 航点列表
        self.n_points = n_points
        self.priority = priority
        self.dt = dt
        self.radius = radius
        self.goal_threshold = goal_threshold
        self.arrive_flag = False
        self.collision_flag = False
        self.destination_arrive_flag = False
        self.reached_goal = False

        # 航点管理（原有逻辑）
        if len(self.waypoints) > 1:
            self.current_des = self.waypoints[1].copy()
            self.previous_des = self.waypoints[0].copy()
        elif len(self.waypoints) == 1:
            self.current_des = self.destination.copy()
            self.previous_des = self.waypoints[0].copy()
        else:
            self.current_des = self.destination.copy()
            self.previous_des = self.starting.copy()
        self.i = 1   # 当前航点索引

        # 噪声参数（用于模拟控制误差）
        self.__noise = kwargs.get('noise', True)
        self.__control_std = kwargs.get('control_std', [0.06, 0.06, 0.06])

        # 轨迹记录
        self.path_history = [self.starting.copy()]
        self.speed_history = [0.0]

        # ========== 新增：航路长度与进度 ==========
        self.route_length = 0.0   # 航路总长度
        self.progress = 0.0       # 从起点开始的累计移动距离
        self.waypoint_progress = 0  # 当前航点进度（用于奖励计算）
        self.min_ttc = None       # 最小TTC（用于奖励计算）
        self._compute_route_length()

    def _compute_route_length(self):
        """根据航点计算航路总长度"""
        if len(self.waypoints) >= 2:
            total = 0.0
            for i in range(len(self.waypoints)-1):
                total += np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
            self.route_length = total
        else:
            self.route_length = 0.0

    def _update_progress(self, speed):
        """根据速度更新进度"""
        if self.route_length > 0:
            self.progress += speed * self.dt
            if self.progress > self.route_length:
                self.progress = self.route_length

    def move(self, vel_cmd):
        """
        根据给定的速度指令更新位置（含可选噪声）。
        （原有逻辑不变，仅添加进度更新）
        """
        # 如果已经到达终点，则保持原地不动
        if self.reached_goal:
            self.vel = np.zeros(3, dtype=float)
            return

        if self.__noise:
            noise = np.random.normal(0, self.__control_std, size=3)
            vel_cmd = vel_cmd + noise
        # 速度大小约束
        speed = np.linalg.norm(vel_cmd)
        if speed > self.max_speed:
            vel_cmd = vel_cmd / speed * self.max_speed
            speed = self.max_speed
        self.vel = vel_cmd
        self.state = self.state + self.vel * self.dt
        self.path_history.append(self.state.copy())
        self.speed_history.append(speed)

        # 新增：更新进度
        self._update_progress(speed)

        # 检查是否到达当前航点（原有逻辑）
        if self.arrive(self.state, self.current_des):
            self._advance_waypoint()
        # 处理最后一个目标可能已被越过的情况
        elif self.i == self.n_points - 1 and np.linalg.norm(self.current_des - self.destination) < 1e-6:
            if np.dot(self.state - self.current_des, self.current_des - self.previous_des) > 0:
                self.state = self.destination.copy()
                self.vel = np.zeros(3, dtype=float)
                self.destination_arrive_flag = True
                self.reached_goal = True

    def _advance_waypoint(self):
        """切换到下一个航点（原有逻辑，未修改）"""
        if self.i < self.n_points - 1:
            self.i += 1
            self.previous_des = self.current_des
            self.current_des = self.waypoints[self.i]
            self.arrive_flag = False
            self.waypoint_progress = self.i  # 更新航点进度
        # 如果到达最后一个航点，则当前目标为最终目的地
        elif self.i == self.n_points - 1:
            if self.arrive(self.state, self.destination):
                self.destination_arrive_flag = True
                self.reached_goal = True

    def arrive(self, pos, target):
        """判断是否到达指定点（考虑阈值）"""
        return np.linalg.norm(pos - target) <= self.goal_threshold

    def cal_des_vel(self):
        """
        计算指向当前目标点的期望速度（用于导航或混合控制）。
        """
        dis, angles = self.relative(self.state, self.current_des)
        if dis > self.goal_threshold:
            dir_vec = self.angles_to_direction(angles)
            # 速度随距离平滑减小
            speed_ratio = min(1.0, dis / (5.0 * self.goal_threshold))
            desired_speed = self.max_speed * max(0.1, speed_ratio)
            return dir_vec * desired_speed
        else:
            return np.zeros(3)

    def reset(self):
        """重置无人机到初始状态"""
        self.state = self.starting.copy()
        self.vel = np.zeros(3)
        self.arrive_flag = False
        self.collision_flag = False
        self.destination_arrive_flag = False
        self.reached_goal = False
        self.i = 1
        self.waypoint_progress = 0  # 重置航点进度
        if len(self.waypoints) > 1:
            self.current_des = self.waypoints[1].copy()
            self.previous_des = self.waypoints[0].copy()
        else:
            self.current_des = self.destination.copy()
            self.previous_des = self.starting.copy()
        self.path_history = [self.starting.copy()]
        self.speed_history = [0.0]

        # 新增：重置进度
        self.progress = 0.0

    # ---------- 静态辅助方法 ----------
    @staticmethod
    def relative(pos1, pos2):
        """计算两点间的距离和方向角（方位角、俯仰角）"""
        delta = np.array(pos2[:3]) - np.array(pos1[:3])
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return 0, (0, 0)
        azimuth = np.arctan2(delta[1], delta[0])
        elevation = np.arctan2(delta[2], np.linalg.norm(delta[:2]))
        return dist, (azimuth, elevation)

    @staticmethod
    def angles_to_direction(angles):
        """将方位角、俯仰角转换为单位方向向量"""
        az, el = angles
        return np.array([
            np.cos(az) * np.cos(el),
            np.sin(az) * np.cos(el),
            np.sin(el)
        ])

    @staticmethod
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    @staticmethod
    def distance2D(p1, p2):
        return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

    # ---------- 碰撞检测 ----------
    def collision_check_with_dro(self, other_drones):
        """检查与列表中其他无人机的碰撞（使用保护圆球）"""
        for other in other_drones:
            if other is self or other.collision_flag:
                continue
            if self.distance(self.state, other.state) <= self.radius + other.radius:
                self.collision_flag = True
                other.collision_flag = True
                return True
        return False

    def collision_check_with_building(self, building_list):
        """检查与建筑物的碰撞（建筑物为 [x, y, h, r]）"""
        for b in building_list:
            if self.state[2] <= b[2]:   # 高度低于建筑物
                if self.distance2D(self.state, (b[0], b[1])) <= self.radius + b[3]:
                    self.collision_flag = True
                    return True
        return False

    # ========== 新增：状态提取方法 ==========
    def dronestate(self):
        """
        返回无人机状态，用于RL训练。
        返回向量 [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, progress, des_vel_x, des_vel_y, des_vel_z]
        """
        des_vel = self.cal_des_vel()   # 期望速度向量
        return np.concatenate([
            self.state,                # 3维
            self.vel,                  # 3维
            [self.progress],           # 1维
            des_vel                    # 3维
        ])

    def obs_state(self):
        """
        返回其他无人机观测该无人机所需的信息，用于RVO等。
        返回向量 [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, radius, type]
        其中 type = 1 表示航路无人机，0 表示自由无人机（由子类设置）。
        """
        # 注意：type 需要在子类中设置 self.uav_type_num，默认为0
        type_val = getattr(self, 'uav_type_num', 0)
        return np.array([
            self.state[0], self.state[1], self.state[2],
            self.vel[0], self.vel[1], self.vel[2],
            self.radius,
            type_val
        ])