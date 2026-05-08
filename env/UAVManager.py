import numpy as np
from env.BaseUAV import BaseUAV
from env.RouteUAV import RouteUAV
from env.FreeUAV import FreeUAV

# ---------- 无人机管理器 UAVManager ----------
class UAVManager:
    """
    无人机管理器，负责创建和管理 RouteUAV 和 FreeUAV，并提供步进、状态提取、重置等功能。
    """
    def __init__(self, route_configs=None, free_configs=None, building_list=None, step_time=1, **kwargs):
        """
        :param route_configs: list of dict, each with keys:
               'id', 'starting', 'destination', 'waypoints', 'n_points', 'priority' (optional)
        :param free_configs: list of dict, each with keys:
               'id', 'starting', 'destination', 'waypoints' (optional), 'n_points' (optional),
               'priority' (optional), 'safety_radius' (optional)
        :param building_list: list of obstacles, each as [x, y, h, r]
        :param step_time: simulation time step
        :param kwargs: additional arguments passed to UAV constructors (e.g., noise, control_std)
        """
        self.route_uavs = []
        self.free_uavs = []
        self.building_list = building_list if building_list else []
        self.step_time = step_time
        self.kwargs = kwargs

        if route_configs:
            for cfg in route_configs:
                uav = RouteUAV(
                    id=cfg['id'],
                    starting=cfg['starting'],
                    destination=cfg['destination'],
                    waypoints=cfg['waypoints'],
                    n_points=cfg['n_points'],
                    init_acc=1.0,   # 可配置
                    priority=cfg.get('priority', 1),
                    dt=step_time,
                    **kwargs
                )
                self.route_uavs.append(uav)

        if free_configs:
            for cfg in free_configs:
                # 如果未提供 waypoints，则使用起点和终点作为默认航点
                waypoints = cfg.get('waypoints', [cfg['starting'], cfg['destination']])
                n_points = cfg.get('n_points', len(waypoints))
                uav = FreeUAV(
                    id=cfg['id'],
                    starting=cfg['starting'],
                    destination=cfg['destination'],
                    waypoints=waypoints,
                    n_points=n_points,
                    init_acc=1.0,
                    priority=cfg.get('priority', 1),
                    dt=step_time,
                    safety_radius=cfg.get('safety_radius', None),
                    **kwargs
                )
                self.free_uavs.append(uav)

    def step(self, route_actions=None):
        """
        执行一步仿真。
        :param route_actions: list of acceleration scalars for each route UAV (length must match len(route_uavs))
                               If None, route UAVs will maintain current velocity (no control).
        """
        # 1. 更新自由无人机
        if self.free_uavs:
            # 预先计算所有自由无人机的速度指令
            free_vel_cmds = []
            for uav in self.free_uavs:
                others = [other for other in self.free_uavs if other is not uav]
                vel = uav.compute_avoidance_velocity(others)
                free_vel_cmds.append(vel)
            # 统一应用速度，并在到达终点后立即重置到起点
            for uav, vel in zip(self.free_uavs, free_vel_cmds):
                uav.move(vel)
                if uav.reached_goal:
                    uav.reset()

        # 2. 更新航路无人机
        if route_actions is not None:
            assert len(route_actions) == len(self.route_uavs), \
                f"route_actions length {len(route_actions)} does not match route UAV count {len(self.route_uavs)}"
            for uav, acc in zip(self.route_uavs, route_actions):
                uav.apply_acceleration(acc)
        else:
            # 如果没有提供动作，则保持当前速度
            # 这里选择保持当前速度，不做任何改变
            pass

    def check_collisions(self):
        """
        检查所有无人机之间及与建筑物的碰撞，并设置 collision_flag。
        返回碰撞对列表，每对为 (id1, id2) 或 (id, 'building')。
        自由无人机的重置在检测循环结束后统一执行，避免循环内重置导致
        collision_flag 清零引发同一帧内重复碰撞计数的问题。
        """
        collisions = []
        route_route_collisions = 0
        route_free_collisions = 0
        route_building_collisions = 0
        all_uavs = self.route_uavs + self.free_uavs
        # 收集需要重置的自由无人机，循环结束后统一处理
        free_uavs_to_reset = set()

        # 无人机间碰撞
        for i, uav in enumerate(all_uavs):
            if uav.collision_flag or uav.destination_arrive_flag:
                continue
            for j, other in enumerate(all_uavs):
                if i <= j:
                    continue
                if other.collision_flag or other.destination_arrive_flag:
                    continue
                if uav.distance(uav.state, other.state) <= uav.radius + other.radius:
                    uav.collision_flag = True
                    other.collision_flag = True
                    collisions.append((uav.id, other.id))
                    if uav in self.route_uavs and other in self.route_uavs:
                        route_route_collisions += 1
                    elif (uav in self.route_uavs and other in self.free_uavs) or \
                         (uav in self.free_uavs and other in self.route_uavs):
                        route_free_collisions += 1
                        if uav in self.free_uavs:
                            free_uavs_to_reset.add(uav)
                        if other in self.free_uavs:
                            free_uavs_to_reset.add(other)

        # 检测循环结束后统一重置碰撞的自由无人机
        for uav in free_uavs_to_reset:
            uav.reset()


        # # 无人机与建筑物碰撞
        # for uav in all_uavs:
        #     if uav.collision_flag:
        #         continue
        #     for b in self.building_list:
        #         if uav.state[2] <= b[2] and uav.distance2D(uav.state, (b[0], b[1])) <= uav.radius + b[3]:
        #             uav.collision_flag = True
        #             collisions.append((uav.id, 'building'))
        #             if uav in self.route_uavs:
        #                 route_building_collisions += 1
                  
        # # 碰撞后重置状态
        # for uav in all_uavs:
        #     if uav.collision_flag:
        #         uav.reset()

        return collisions, route_route_collisions, route_free_collisions, route_building_collisions

    def all_route_arrived(self):
        """检查所有航路无人机是否到达目标点"""
        for uav in self.route_uavs:
            if not uav.reached_goal:
                return False
        return True

    def all_arrived(self):
        """兼容旧接口：检查所有航路无人机是否到达目标点"""
        return self.all_route_arrived()

    def reset(self):
        """重置所有无人机到初始状态"""
        for uav in self.route_uavs:
            uav.reset()
        for uav in self.free_uavs:
            uav.reset()

    def get_route_states(self):
        """返回所有航路无人机的完整状态列表（用于训练）"""
        return [uav.dronestate() for uav in self.route_uavs]

    def get_free_states(self):
        """返回所有自由无人机的简约状态列表（用于其他无人机观测）"""
        return [uav.obs_state() for uav in self.free_uavs]

    def get_all_uavs(self):
        """返回所有无人机的合并列表"""
        return self.route_uavs + self.free_uavs

    def get_num_route(self):
        return len(self.route_uavs)

    def get_num_free(self):
        return len(self.free_uavs)

    def get_route_uavs(self):
        return self.route_uavs

    def get_free_uavs(self):
        return self.free_uavs
    




# # test_manager.py

# def test_uav_manager():
#     # 定义两架航路无人机的配置
#     route_configs = [
#         {
#             'id': 0,
#             'starting': [0, 0, 0],
#             'destination': [100, 0, 0],
#             'waypoints': [[0, 0, 0], [50, 0, 0], [100, 0, 0]],
#             'n_points': 3,
#             'priority': 1
#         },
#         {
#             'id': 1,
#             'starting': [0, 20, 0],
#             'destination': [100, 20, 0],
#             'waypoints': [[0, 20, 0], [50, 20, 0], [100, 20, 0]],
#             'n_points': 3,
#             'priority': 1
#         }
#     ]

#     # 定义三架自由无人机的配置
#     free_configs = [
#         {
#             'id': 2,
#             'starting': [50, -20, 0],
#             'destination': [50, 80, 0],
#             'waypoints': [[50, -20, 0], [50, 80, 0]],   # 直线
#             'n_points': 2,
#             'priority': 2,
#             'safety_radius': 2.0
#         },
#         {
#             'id': 3,
#             'starting': [10, 10, 0],
#             'destination': [90, 90, 0],
#             'waypoints': [[10, 10, 0], [90, 90, 0]],
#             'n_points': 2,
#             'priority': 2,
#             'safety_radius': 2.0
#         },
#         {
#             'id': 4,
#             'starting': [90, -10, 0],
#             'destination': [20, 70, 0],
#             'waypoints': [[90, -10, 0], [20, 70, 0]],
#             'n_points': 2,
#             'priority': 2,
#             'safety_radius': 2.0
#         }
#     ]
