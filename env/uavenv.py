import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from env.UAVManager import UAVManager
from env.ScenarioGeneratorV2 import ScenarioGeneratorV2
from env.ScenarioGeneratorV3 import ScenarioGeneratorV3
from env.RewardCalculator import RewardCalculator
from env.ObservationNormalizer import ObservationNormalizer  # 【改进】导入归一化器
from config import Config  # 【改进】导入配置

class UAVEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 num_route_uavs=5,
                 num_free_uavs=10,
                 scenario_generator=None,
                 dt=1.0,
                 max_steps=500,
                 render_mode='human',
                 use_random_scene=False,
                 max_circle_diameter=100,
                 circle_diameters=None,
                 debug_mode=False,  # [fix] 调试模式：在 info 中输出 per-agent 统计
                 **kwargs):
        """
        UAV 多智能体环境

        Args:
            use_random_scene: 是否使用随机场景（同心圆）
            max_circle_diameter: 最大圆直径
            circle_diameters: 各圆直径列表
            debug_mode: 是否启用 per-agent 调试统计
        """
        super().__init__()
        self.num_route_uavs = num_route_uavs
        self.num_free_uavs = num_free_uavs
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.render_mode = render_mode
        self.debug_mode = debug_mode  # [fix] 调试模式开关
        self.kwargs = kwargs
        
        # 随机场景配置
        self.use_random_scene = use_random_scene
        self.max_circle_diameter = max_circle_diameter
        self.circle_diameters = circle_diameters

        self.max_free_uavs = 20          # 最大自由无人机数量
        self.free_obs_dim = 8            # 每个自由无人机的基础特征维度（位置3+速度3+半径1+类型1），TTC后续添加
        self.self_state_dim = 10         # 航路无人机自身状态维度

        # 创建场景生成器（根据 Config.SCENARIO_VERSION 选择版本）
        if scenario_generator is not None:
            self.scenario_generator = scenario_generator
        elif Config.SCENARIO_VERSION == 'v3':
            self.scenario_generator = ScenarioGeneratorV3(
                num_route_uavs,
                num_free_uavs,
                use_random_scene=use_random_scene,
                max_circle_diameter=max_circle_diameter,
                circle_diameters=circle_diameters
            )
        else:  # default: v2
            self.scenario_generator = ScenarioGeneratorV2(
                num_route_uavs,
                num_free_uavs,
                use_random_scene=use_random_scene,
                max_circle_diameter=max_circle_diameter,
                circle_diameters=circle_diameters
            )

        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(num_route_uavs,), dtype=np.float32)
        # 观测空间：每个航路无人机一个观测向量
        # 10(self) + 16(route_threat: 8rel+ttc+prog+type+align+4conflict_onehot) + 5×10(free: 8rel+ttc+cross) = 76
        obs_dim = self.self_state_dim + (self.self_state_dim + 6) + 5 * (self.free_obs_dim + 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_route_uavs, obs_dim),
            dtype=np.float32)

        self.manager = None
        self.prev_progress = None
        self.prev_actions = None
        self._immune_steps: dict = {}   # uav.id → 剩余免疫步数（reset后保护期）

        # 【改进】观测归一化器
        self.normalizer = None
        if Config.NORMALIZE_OBSERVATIONS:
            self.normalizer = ObservationNormalizer(obs_dim=obs_dim, method='fixed')

        # 渲染相关变量
        self.fig = None
        self.ax = None
        self.plots = {}          # 存储每个无人机的绘图对象
        self.static_elements = [] # 存储静态元素（航路、起终点）


    def reset(self, seed=None, options=None, regenerate_scene=True):
        super().reset(seed=seed)

        # 仅在需要时生成新场景（每epoch一次），episode内复位复用同一场景
        if regenerate_scene or self.manager is None:
            route_configs, free_configs = self.scenario_generator.generate()
            self._last_route_configs = route_configs
            self._last_free_configs = free_configs
        else:
            route_configs = self._last_route_configs
            free_configs = self._last_free_configs

        self.manager = UAVManager(
            route_configs=route_configs,
            free_configs=free_configs,
            building_list=[],
            step_time=self.dt,
            **self.kwargs
        )

        self.step_count = 0
        self.prev_progress = [uav.progress for uav in self.manager.get_route_uavs()]
        self.prev_actions = np.zeros(self.num_route_uavs, dtype=np.float32)
        self.episode_punished_ids = set()   # 每回合重置已惩罚ID集合
        self._immune_steps = {}             # 每回合清空免疫计数器

        # 重置渲染图形（如果启用）
        if self.render_mode == 'human' and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.plots.clear()

        obs_list, _ = self._get_obs()  # [fix] _get_obs 现在返回 (obs_list, ttc_info)
        info = {}
        return obs_list, info

    @staticmethod
    def _ego_heading(route):
        """返回UAV当前航向的水平单位向量和侧向单位向量。"""
        h2d = route.vel[:2].copy()
        h2d_norm = float(np.linalg.norm(h2d))
        if h2d_norm < 0.1:
            d = route.current_des[:2] - route.state[:2]
            d_norm = float(np.linalg.norm(d))
            h2d = d / d_norm if d_norm > 1e-6 else np.array([1.0, 0.0])
        else:
            h2d = h2d / h2d_norm
        lat2d = np.array([-h2d[1], h2d[0]])
        return h2d, lat2d

    @staticmethod
    def _rel_features(route, other_state, other_vel, h2d, lat2d):
        """
        计算 route 视角下 other 的相对特征（8个标量）：
        rel_pos_fwd, rel_pos_lat, rel_pos_alt, dist_3d,
        other_speed, rel_vel_fwd, rel_vel_lat, rel_vel_alt
        """
        rel_pos = other_state[:3] - route.state[:3]
        rel_pos_fwd  = float(np.dot(rel_pos[:2], h2d))
        rel_pos_lat  = float(np.dot(rel_pos[:2], lat2d))
        rel_pos_alt  = float(rel_pos[2])
        dist_3d      = float(np.linalg.norm(rel_pos))
        other_speed  = float(np.linalg.norm(other_vel))
        rv = route.vel - other_vel
        rel_vel_fwd  = float(np.dot(rv[:2], h2d))
        rel_vel_lat  = float(np.dot(rv[:2], lat2d))
        rel_vel_alt  = float(rv[2])
        return rel_pos_fwd, rel_pos_lat, rel_pos_alt, dist_3d, \
               other_speed, rel_vel_fwd, rel_vel_lat, rel_vel_alt

    @staticmethod
    def _batch_ttc(pos_a, pos_b, vel_a, vel_b, ra, rb):
        """向量化 TTC 计算，返回 (n, m) 矩阵（秒）。"""
        # (n,1,3) - (1,m,3) → (n,m,3)
        rp = pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :]
        rv = vel_a[:, np.newaxis, :] - vel_b[np.newaxis, :, :]
        r  = ra[:, np.newaxis] + rb[np.newaxis, :]          # (n,m)

        a = np.sum(rv ** 2,      axis=-1)                   # (n,m)
        b = 2.0 * np.sum(rp * rv, axis=-1)
        c = np.sum(rp ** 2,      axis=-1) - r ** 2

        ttc = np.full(a.shape, 1e6, dtype=np.float32)
        ttc[c <= 0] = 0.0                                   # 已在碰撞范围内

        valid = (a > 1e-10) & (c > 0)
        disc  = np.where(valid, b ** 2 - 4.0 * a * c, -1.0)
        valid &= (disc >= 0)

        if valid.any():
            sq     = np.sqrt(np.maximum(disc, 0.0))
            safe_a = np.where(a > 1e-10, 2.0 * a, 1.0)   # 避免 np.where 预求值时除零
            t1  = np.where(valid, (-b - sq) / safe_a, 1e6)
            t2  = np.where(valid, (-b + sq) / safe_a, 1e6)
            t1  = np.where(t1 >= 0, t1, 1e6)
            t2  = np.where(t2 >= 0, t2, 1e6)
            ttc = np.where(valid, np.minimum(t1, t2), ttc)

        return ttc.astype(np.float32)

    def _get_obs(self):
        """
        构建相对坐标观测（自我中心表示）。
        维度：10(self) + 16(route_threat: 8rel+ttc+prog+type+align+4onehot) + 5×10(free) = 76。
        TTC 采用向量化批量计算，比循环快 5-10 倍。
        """
        route_uavs = self.manager.get_route_uavs()
        free_uavs  = self.manager.get_free_uavs()
        n = len(route_uavs)
        obs_list = []

        # [fix] 显式存储 TTC 信息为返回值，不再写入 UAV 对象的临时属性
        ttc_info = {
            'min_ttc_route': [None] * n,
            'min_ttc_free': [None] * n,
            'min_ttc': [None] * n,
            'ttc_route_lateral': [None] * n,
        }

        # ── 预计算所有 TTC（向量化）─────────────────────────────────
        r_pos = np.array([u.state  for u in route_uavs], dtype=np.float32)  # (n,3)
        r_vel = np.array([u.vel    for u in route_uavs], dtype=np.float32)
        r_rad = np.array([u.radius for u in route_uavs], dtype=np.float32)

        rr_ttc = self._batch_ttc(r_pos, r_pos, r_vel, r_vel, r_rad, r_rad)  # (n,n)
        np.fill_diagonal(rr_ttc, 1e6)  # 排除自身

        if free_uavs:
            f_pos = np.array([u.state  for u in free_uavs], dtype=np.float32)
            f_vel = np.array([u.vel    for u in free_uavs], dtype=np.float32)
            f_rad = np.array([u.radius for u in free_uavs], dtype=np.float32)
            rf_ttc = self._batch_ttc(r_pos, f_pos, r_vel, f_vel, r_rad, f_rad)  # (n,m)
        else:
            rf_ttc = np.full((n, 0), 1e6, dtype=np.float32)

        for i, route in enumerate(route_uavs):
            # ── 自身状态 (10维) ──────────────────────────────────────
            speed     = float(np.linalg.norm(route.vel))
            des_vel   = route.cal_des_vel()
            des_speed = float(np.linalg.norm(des_vel))
            dist_to_wp = float(np.linalg.norm(route.current_des - route.state))
            prog_ratio = float(np.clip(route.progress / max(route.route_length, 1e-6), 0.0, 1.0))

            if speed > 0.1:
                heading = route.vel / speed
            else:
                d = route.current_des - route.state
                dn = np.linalg.norm(d)
                heading = d / dn if dn > 1e-6 else np.array([1.0, 0.0, 0.0])

            des_heading = des_vel / des_speed if des_speed > 0.1 else heading.copy()

            self_state = np.array([
                speed, des_speed, dist_to_wp, prog_ratio,
                heading[0], heading[1], heading[2],
                des_heading[0], des_heading[1], des_heading[2]
            ], dtype=np.float32)

            h2d, lat2d = self._ego_heading(route)

            # ── 最威胁航路UAV (16维) ─────────────────────────────────
            other_route_obs = np.zeros(self.self_state_dim + 6, dtype=np.float32)
            best_ttc_route  = float(np.min(rr_ttc[i]))
            best_j_rr       = int(np.argmin(rr_ttc[i]))

            if best_ttc_route < 1e6:
                best_other = route_uavs[best_j_rr]
                f = self._rel_features(route, best_other.state, best_other.vel, h2d, lat2d)
                op = float(np.clip(best_other.progress / max(best_other.route_length, 1e-6), 0, 1))

                # ── 改进1：航向对齐度（替换原来的 0.0 占位符）──────────
                # [-1, 1]：+1=同向, 0=垂直交叉, -1=正面对冲
                # 让策略区分"正面对冲（需要一让一走）"和"追尾/侧交叉"
                other_vel_2d = best_other.vel[:2]
                other_spd_2d = np.linalg.norm(other_vel_2d)
                if other_spd_2d > 0.1:
                    other_h2d_obs = other_vel_2d / other_spd_2d
                else:
                    od = best_other.current_des[:2] - best_other.previous_des[:2]
                    odn = np.linalg.norm(od)
                    other_h2d_obs = od / odn if odn > 1e-6 else np.array([1.0, 0.0])
                heading_align = float(np.dot(h2d, other_h2d_obs))  # [-1, 1]

                # ── 新增：冲突类型 4 维 one-hot（索引12-15）─────────────
                # class 0: 同向追尾 (align>0.7 且 对方在前方 rel_pos_fwd>0)
                # class 1: 正面对冲 (align<-0.7)
                # class 2: 侧向交叉 (|align|<=0.2)
                # class 3: 斜向交叉 (其他情况)
                # rel_pos_fwd = f[0]: >0 对方在前方, <0 对方在后方
                if heading_align > 0.7 and f[0] > 0:
                    conflict_type = 0
                elif heading_align < -0.7:
                    conflict_type = 1
                elif abs(heading_align) <= 0.2:
                    conflict_type = 2
                else:
                    conflict_type = 3
                conflict_onehot = np.zeros(4, dtype=np.float32)
                conflict_onehot[conflict_type] = 1.0

                other_route_obs = np.array([
                    f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
                    float(np.clip(best_ttc_route, 0, 999)), op,
                    0.0,            # [10] type indicator: route=0, free=1
                    heading_align,  # [11] heading alignment: +1=same-dir, -1=head-on
                    conflict_onehot[0],  # [12] conflict class 0: 同向追尾
                    conflict_onehot[1],  # [13] conflict class 1: 正面对冲
                    conflict_onehot[2],  # [14] conflict class 2: 侧向交叉
                    conflict_onehot[3],  # [15] conflict class 3: 斜向交叉
                ], dtype=np.float32)

                # ── 改进2：侧向优先权信号（供奖励函数打破对称性）───────
                # rel_pos_lat > 0 → other 在我左侧 → 我是右侧来车 → 我有优先权
                # rel_pos_lat < 0 → other 在我右侧 → 我是左侧来车 → 我应让行
                ttc_info['ttc_route_lateral'][i] = f[1]  # [fix] 写入 ttc_info 而非 uav 属性
            else:
                ttc_info['ttc_route_lateral'][i] = None

            # ── 最威胁自由UAV × 5 (10维/个) ─────────────────────────
            free_obs      = np.zeros((5, self.free_obs_dim + 2), dtype=np.float32)
            best_ttc_free = 1e6

            if rf_ttc.shape[1] > 0:
                sorted_free = np.argsort(rf_ttc[i])
                best_ttc_free = float(rf_ttc[i, sorted_free[0]])
                for k, j in enumerate(sorted_free[:5]):
                    ttc  = float(rf_ttc[i, j])
                    free = free_uavs[j]
                    f    = self._rel_features(route, free.state, free.vel, h2d, lat2d)

                    # ── 路径穿越时间（与TTC并列，不合并）───────────────
                    # 自由UAV侧向穿越我飞行轴线的预测时刻；
                    # 即使不引发碰撞（TTC=inf），穿越行为也构成威胁。
                    # 与TTC分开保留，让策略自行学习两者的权重。
                    lat_pos, lat_vel = f[1], f[6]
                    fwd_pos          = f[0]
                    path_cross_t     = 1e6
                    if lat_pos * lat_vel > 0 and abs(lat_vel) > 0.01:
                        lat_cross_t = lat_pos / lat_vel
                        if 0 < lat_cross_t < 30:              # 30s内会穿越中心线
                            fwd_at_cross = fwd_pos - f[5] * lat_cross_t
                            if abs(fwd_at_cross) < 25:        # 穿越时距我前方25m内
                                path_cross_t = lat_cross_t

                    free_obs[k] = np.array([
                        f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
                        float(np.clip(ttc, 0, 999)),           # [8] TTC
                        float(np.clip(path_cross_t, 0, 999))   # [9] path crossing time
                    ], dtype=np.float32)

            # [fix] 存储分类 TTC 到显式字典，供奖励计算和冲突统计
            ttc_info['min_ttc_route'][i] = best_ttc_route if best_ttc_route < 1e6 else None
            ttc_info['min_ttc_free'][i]  = best_ttc_free  if best_ttc_free  < 1e6 else None
            mn = min(best_ttc_route, best_ttc_free)
            ttc_info['min_ttc'][i] = mn if mn < 1e6 else None

            # ── 拼接 & 归一化 ───────────────────────────────────────
            obs = np.concatenate([self_state, other_route_obs, free_obs.flatten()])
            if not np.isfinite(obs).all():
                obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

            if hasattr(self, 'normalizer') and self.normalizer is not None:
                obs = self.normalizer.normalize(obs)

            obs_list.append(obs.astype(np.float32))

        return obs_list, ttc_info  # [fix] 显式返回 TTC 信息，消除副作用

    def step(self, actions):
        RESET_IMMUNITY_STEPS = 3   # reset后3步内忽略碰撞和TTC惩罚

        reached_goals=[]
        arrive_flags=[]

        # 递减免疫计数器，移除已到期的条目
        self._immune_steps = {uid: cnt - 1
                              for uid, cnt in self._immune_steps.items() if cnt > 1}
        immune_ids = set(self._immune_steps.keys())

        prev_progress = [uav.progress for uav in self.manager.get_route_uavs()]

        self.manager.step(route_actions=actions)
        reached_goals = [uav.reached_goal for uav in self.manager.route_uavs]
        arrive_flags = [uav.destination_arrive_flag for uav in self.manager.route_uavs]

        all_arrived = min(arrive_flags) if arrive_flags else False  # 只有当所有 UAV 都到达目的地时才算全部到达

        collisions, route_route_collisions, route_free_collisions, route_building_collisions = self.manager.check_collisions()

        # 提取航路UAV中受到碰撞的ID列表，免疫中的UAV不处理
        route_uavs = self.manager.get_route_uavs()
        route_uav_ids = {uav.id for uav in route_uavs}
        collision_route_uav_ids = set()
        for uav1_id, uav2_id in collisions:
            if uav1_id in route_uav_ids and uav1_id not in immune_ids:
                collision_route_uav_ids.add(uav1_id)
            if uav2_id in route_uav_ids and uav2_id not in immune_ids:
                collision_route_uav_ids.add(uav2_id)

        # 收集到达终点的航路无人机 ID
        arrived_route_uav_ids = set()

        for uav in self.manager.route_uavs:
            if uav.reached_goal and not uav.collision_flag:
                arrived_route_uav_ids.add(uav.id)

        # [fix] 先调用 _get_obs() 获取观测和 TTC 信息，然后统计冲突和计算奖励
        obs_list, ttc_info = self._get_obs()

        # 免疫UAV的TTC信号清零，避免不可学习的TTC惩罚
        for idx, uav in enumerate(route_uavs):
            if uav.id in immune_ids:
                ttc_info['min_ttc_route'][idx] = None
                ttc_info['min_ttc_free'][idx]  = None
                ttc_info['min_ttc'][idx]       = None
                ttc_info['ttc_route_lateral'][idx] = None

        # 分类统计 TTC<3s 冲突（Route-Route / Route-Free 分开计数，跳过免疫UAV）
        ttc_threshold_conflict = 3.0
        rr_conflict_count = 0
        rf_conflict_count = 0
        for idx, uav in enumerate(route_uavs):
            if uav.id in immune_ids:
                continue
            if ttc_info['min_ttc_route'][idx] is not None \
                    and ttc_info['min_ttc_route'][idx] < ttc_threshold_conflict:
                rr_conflict_count += 1
            if ttc_info['min_ttc_free'][idx] is not None \
                    and ttc_info['min_ttc_free'][idx] < ttc_threshold_conflict:
                rf_conflict_count += 1
        conflict_count = rr_conflict_count + rf_conflict_count

        # 获取航点数量（假设所有route UAV有相同的航点数）
        num_waypoints = len(route_uavs[0].waypoints) if route_uavs else 5

        # [fix] 提前构建重置掩码，供 RewardCalculator 的 reset_mask 参数
        per_agent_done = [False] * self.num_route_uavs
        for idx, uav in enumerate(route_uavs):
            if uav.id in collision_route_uav_ids or uav.id in arrived_route_uav_ids:
                per_agent_done[idx] = True

        # 计算每个航路无人机的奖励
        # 【改进】从config导入奖励参数
        from config import Config
        # [fix] 从 RouteUAV 读取实际最大速度，替代硬编码 2.0
        real_max_speed = route_uavs[0].max_speed if route_uavs else 2.0
        rewards = RewardCalculator.compute(
            route_uavs,
            prev_progress=prev_progress,
            prev_actions=self.prev_actions,
            current_actions=actions,  # 传入当前动作
            reached_goals=reached_goals,   # 新增传入
            arrive_flags=arrive_flags,     # 新增传入
            collision_uav_ids=collision_route_uav_ids,  # 【改进】传递具体的碰撞UAV ID
            punished_collision_ids=self.episode_punished_ids,   # 【改进】传递本回合已经惩罚过的ID集合
            max_speed=real_max_speed,  # [fix] 使用 UAV 实际最大速度
            waypoint_rewards=None,  # 【改进】不再使用固定列表
            num_waypoints=num_waypoints,  # 【改进】传递航点数量
            ttc_threshold=5.0,  # TTC阈值
            collision_penalty=Config.COLLISION_PENALTY,  # 【改进】从config获取
            waypoint_coef=Config.WAYPOINT_REWARD_COEF,  # 【改进】从config获取
            terminal_coef=Config.TERMINAL_REWARD_COEF,  # 【改进】从config获取
            progress_scale=Config.PROGRESS_REWARD_SCALE,  # 【改进】从config获取
            # [fix] 显式传递 TTC 信息，替代从 uav 属性隐式读取
            min_ttc_route_list=ttc_info['min_ttc_route'],
            min_ttc_free_list=ttc_info['min_ttc_free'],
            ttc_route_lateral_list=ttc_info['ttc_route_lateral'],
            reset_mask=per_agent_done,  # [fix] 传递重置掩码以跳过平滑奖励
        )

        self.prev_actions = actions.copy()
        self.step_count += 1

        out_of_bounds = self._is_any_uav_out_of_bounds()
        terminated = all_arrived or out_of_bounds
        truncated = self.step_count >= self.max_steps

        # 单架无人机碰撞或到达时：重置该无人机，清除其惩罚记录，
        # 并将其 prev_actions 归零，使其以干净状态开始新轨迹。
        for idx, uav in enumerate(route_uavs):
            if per_agent_done[idx]:
                uav.reset()
                self.episode_punished_ids.discard(uav.id)
                self.prev_actions[idx] = 0.0
                self.prev_progress[idx] = 0.0  # [fix] 避免跨重置的虚假进度跳跃
                self._immune_steps[uav.id] = RESET_IMMUNITY_STEPS  # 授予碰撞免疫

        # 若有无人机被重置，重新获取观测（返回重置后的起点观测）
        if any(per_agent_done):
            obs_list, _ = self._get_obs()  # [fix] _get_obs 返回元组

        info = {
            'collisions': collisions,
            'route_route_collisions': route_route_collisions,
            'route_free_collisions': route_free_collisions,
            'route_building_collisions': route_building_collisions,
            'out_of_bounds': out_of_bounds,
            'step': self.step_count,
            'all_arrived': all_arrived,
            'conflict_count': conflict_count,
            'rr_conflict_count': rr_conflict_count,
            'rf_conflict_count': rf_conflict_count,
            'collision_route_uav_ids': collision_route_uav_ids,
            'arrived_route_uav_ids': arrived_route_uav_ids,
            'per_agent_done': per_agent_done,  # 单架无人机轨迹终止标志
        }

        # [fix] 调试模式：输出 per-agent 的 TTC 和奖励分解
        if self.debug_mode:
            info['per_agent_min_ttc'] = list(ttc_info['min_ttc'])
            info['per_agent_reward'] = list(rewards)

        # 渲染（如果启用）
        if self.render_mode == 'human':
            self.render()

        # print(f"Observation: {obs_list}")
        # print(f"Reward: {rewards}")
        # print(f"Terminated: {terminated}")
        # print(f"Truncated: {truncated}")
        # print(f"Info: {info}")

        return obs_list, rewards, terminated, truncated, info

    def _is_any_uav_out_of_bounds(self):
        """检查任意 UAV 是否飞出场景边界（以超大圆+缓冲为边界，距场景中心计算）。"""
        gen = self.scenario_generator
        boundary_radius = gen.xlarge_diameter / 2.0 + 10.0   # 超大圆半径 + 10m 缓冲
        cx, cy = gen.center[0], gen.center[1]
        for uav in self.manager.get_all_uavs():
            if np.linalg.norm([uav.state[0] - cx, uav.state[1] - cy]) > boundary_radius:
                return True
        return False

    def render(self):
        """
        使用 matplotlib 实时渲染所有无人机位置、航路和起终点。
        """
        if self.render_mode != 'human':
            return

        # 初始化图形（第一次调用时）
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.ax.set_aspect('equal')
            # 动态坐标范围：以场景中心为基准，场景半径+15m边距
            gen = self.scenario_generator
            cx, cy = gen.center[0], gen.center[1]
            margin = 15
            r = gen.scene_radius + margin
            self.ax.set_xlim(cx - r, cx + r)
            self.ax.set_ylim(cy - r, cy + r)
            self.ax.grid(True)
            self.ax.set_title(f'UAV Simulation (center=({cx:.0f},{cy:.0f}), r={gen.scene_radius:.0f}m)')

            # 绘制所有航路无人机的航路（静态）
            for uav in self.manager.get_route_uavs():
                waypoints = uav.waypoints
                xs = [wp[0] for wp in waypoints]
                ys = [wp[1] for wp in waypoints]
                line, = self.ax.plot(xs, ys, 'b--', alpha=0.5, linewidth=1)
                self.static_elements.append(line)

            # 绘制所有无人机的起点和终点（静态）
            for uav in self.manager.get_all_uavs():
                start_point, = self.ax.plot(uav.starting[0], uav.starting[1], 'go', markersize=8, alpha=0.7)
                end_point, = self.ax.plot(uav.destination[0], uav.destination[1], 'r*', markersize=10, alpha=0.7)
                self.static_elements.append(start_point)
                self.static_elements.append(end_point)

            # 初始化动态点：每个无人机一个点
            for uav in self.manager.get_all_uavs():
                point, = self.ax.plot([], [], 'o', markersize=6, label=f'UAV {uav.id}')
                self.plots[uav.id] = point

            self.ax.legend(loc='upper right', fontsize='small')
            backend = plt.get_backend().lower()
            if backend not in ['agg', 'template']:
                plt.ion()
                plt.show(block=False)
                plt.pause(0.01)

        # 更新所有无人机位置
        for uav in self.manager.get_all_uavs():
            self.plots[uav.id].set_data([uav.state[0]], [uav.state[1]])

        # 刷新画布
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if plt.get_backend().lower() not in ['agg', 'template']:
            plt.pause(0.01)


    def reset_one_route_uav(self, uav_id):
        """重置单个航路无人机（用于碰撞后重置）"""
        for uav in self.manager.route_uavs:
            if uav.id == uav_id:
                uav.reset()
                break

    def reset_one_free_uav(self, uav_id):
        """重置单个自由无人机"""
        for uav in self.manager.free_uavs:
            if uav.id == uav_id:
                uav.reset()
                break

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def calc_ttc(self, pos_a, pos_b, vel_a, vel_b, ra, rb):
        """计算期望碰撞时间"""
        import math
        rel_x = pos_a[0] - pos_b[0]
        rel_y = pos_a[1] - pos_b[1]
        rel_z = pos_a[2] - pos_b[2]
        rel_vx = vel_a[0] - vel_b[0]
        rel_vy = vel_a[1] - vel_b[1]
        rel_vz = vel_a[2] - vel_b[2]
        r = ra + rb

        a = rel_vx**2 + rel_vy**2 + rel_vz**2
        b = 2 * (rel_x*rel_vx + rel_y*rel_vy + rel_z*rel_vz)
        c = rel_x**2 + rel_y**2 + rel_z**2 - r**2

        # 已经碰撞或非常接近
        if c <= 0:
            return 0.0

        # 相对速度为零，永远不会碰撞（或无穷远）
        if a == 0:
            return 1e6

        delta = b**2 - 4*a*c
        # 无实数解，不会碰撞
        if delta < 0:
            return 1e6

        sqrt_delta = math.sqrt(delta)
        t1 = (-b - sqrt_delta) / (2*a)
        t2 = (-b + sqrt_delta) / (2*a)

        # 取最小非负解
        t = 1e6   # 默认大数
        if t1 >= 0:
            t = t1
        if t2 >= 0 and t2 < t:
            t = t2

        # 确保返回值不超过大数（理论上 t 已经是正数且小于 inf）
        return t if t < 1e6 else 1e6