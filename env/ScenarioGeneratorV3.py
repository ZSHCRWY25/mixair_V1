"""
场景生成器 V3 - 单圆场景
所有无人机（航路 + 自由）均从同一个大圆出发，适用于研究 UAV 数量对解脱成功率的影响。

路线结构（6航点）：
  大圆(进入) → 中圆 → 小圆 → 小圆(转向) → 中圆 → 大圆(飞出)
"""

import numpy as np
from env.RouteGenerator import RouteGenerator


class ScenarioGeneratorV3:
    """
    单圆场景生成器 - 所有UAV在同一圆上出发。

    与 V2 的区别：
    - 无超大圆，航路UAV和自由UAV共用同一个大圆（最大化交通密度）
    - 统一高度范围（所有UAV同层，最大化冲突机会）
    - 6航点路线（较V2的8航点更短，episode步数更少）
    - 适用于研究 UAV 密度对冲突解脱成功率的影响
    """

    def __init__(self, num_route, num_free, route_generator=None,
                 max_circle_diameter=100, circle_diameters=None, scene_seed=None,
                 use_random_scene=False,
                 z_min=20, z_max=25,
                 center=None,
                 **kwargs):
        self.num_route = num_route
        self.num_free = num_free
        self.route_generator = route_generator or RouteGenerator()
        self.max_circle_diameter = max_circle_diameter
        self.scene_seed = scene_seed

        if circle_diameters is None:
            self.circle_diameters = [
                max_circle_diameter,
                max_circle_diameter * 0.7,
                max_circle_diameter * 0.4
            ]
        else:
            self.circle_diameters = circle_diameters

        # V3 没有超大圆；xlarge_diameter 设为大圆直径（供 env 统一接口使用）
        self.xlarge_diameter = max_circle_diameter

        # 统一高度范围（所有UAV同层）
        self.z_min = z_min
        self.z_max = z_max
        self.route_z_min = z_min
        self.route_z_max = z_max
        self.free_z_min = z_min
        self.free_z_max = z_max

        # 场景中心偏移（使所有坐标为正）
        cx, cy = (center[0], center[1]) if center is not None else (100.0, 100.0)
        self.center = np.array([cx, cy, 0.0], dtype=float)

    @property
    def scene_radius(self):
        """场景最大半径（用于渲染边界计算）"""
        return self.max_circle_diameter / 2.0

    def _generate_spiral_route(self, start_angle_offset=None):
        """
        生成6航点螺旋路线：大圆 → 中圆 → 小圆 → 小圆(转向) → 中圆 → 大圆
        """
        big_radius = self.circle_diameters[0] / 2.0
        mid_radius = self.circle_diameters[1] / 2.0
        small_radius = self.circle_diameters[2] / 2.0

        angle_span_deg = np.random.uniform(120, 160)
        angle_span = np.deg2rad(angle_span_deg)

        angle1 = start_angle_offset if start_angle_offset is not None else np.random.uniform(0, 2 * np.pi)
        angle2 = angle1 + angle_span

        z_lo, z_hi = self.z_min, self.z_max
        z_pattern = np.random.randint(0, 3)
        if z_pattern == 0:
            base_z = np.random.uniform(z_lo, z_hi)
            z_vals = [np.clip(base_z + np.random.uniform(-1, 1), z_lo, z_hi) for _ in range(6)]
        elif z_pattern == 1:
            s = np.random.uniform(z_lo, z_hi)
            e = np.random.uniform(z_lo, z_hi)
            z_vals = [s + (e - s) * f for f in [0.0, 0.2, 0.5, 0.5, 0.8, 1.0]]
        else:
            z_vals = [np.random.uniform(z_lo, z_hi) for _ in range(6)]

        max_scale = (self.max_circle_diameter / 2.0) / big_radius
        scale = np.random.uniform(0.95, min(1.05, max_scale))
        big_r = big_radius * scale
        mid_r = mid_radius * scale
        small_r = small_radius * scale

        cx, cy = self.center[0], self.center[1]
        waypoints = [
            np.array([cx + big_r   * np.cos(angle1), cy + big_r   * np.sin(angle1), z_vals[0]], dtype=float),
            np.array([cx + mid_r   * np.cos(angle1), cy + mid_r   * np.sin(angle1), z_vals[1]], dtype=float),
            np.array([cx + small_r * np.cos(angle1), cy + small_r * np.sin(angle1), z_vals[2]], dtype=float),
            np.array([cx + small_r * np.cos(angle2), cy + small_r * np.sin(angle2), z_vals[3]], dtype=float),
            np.array([cx + mid_r   * np.cos(angle2), cy + mid_r   * np.sin(angle2), z_vals[4]], dtype=float),
            np.array([cx + big_r   * np.cos(angle2), cy + big_r   * np.sin(angle2), z_vals[5]], dtype=float),
        ]

        return waypoints, {
            'start_angle': angle1,
            'angle_span': angle_span_deg,
            'z_pattern': z_pattern,
            'scale_factor': scale
        }

    def _validate_starting_positions(self, route_configs, free_configs, min_dist=5.0):
        starts = [cfg['starting'] for cfg in route_configs] + \
                 [cfg['starting'] for cfg in free_configs]
        for i in range(len(starts)):
            for j in range(i + 1, len(starts)):
                if np.linalg.norm(starts[i] - starts[j]) < min_dist:
                    return False
        return True

    def generate(self):
        MIN_START_DIST = 5.0   # 半径1.2×2=2.4m碰撞距离 + 2.6m保护区
        MAX_RETRIES = 300

        big_radius = self.circle_diameters[0] / 2.0
        cx, cy = self.center[0], self.center[1]

        for _ in range(MAX_RETRIES):
            route_configs = []
            for i in range(self.num_route):
                waypoints, meta = self._generate_spiral_route()
                route_configs.append({
                    'id': i,
                    'starting': waypoints[0],
                    'destination': waypoints[-1],
                    'waypoints': waypoints,
                    'n_points': len(waypoints),
                    'priority': 1,
                    'type': 'spiral_v3',
                    'start_angle': meta['start_angle'],
                    'turning_angle': meta['angle_span'],
                })

            free_configs = []
            for i in range(self.num_free):
                a1 = np.random.uniform(0, 2 * np.pi)
                z1 = np.random.uniform(self.z_min, self.z_max)
                start = np.array([cx + big_radius * np.cos(a1),
                                  cy + big_radius * np.sin(a1), z1], dtype=float)
                a2 = a1 + np.random.uniform(np.pi - np.pi / 4, np.pi + np.pi / 4)
                z2 = np.random.uniform(self.z_min, self.z_max)
                end = np.array([cx + big_radius * np.cos(a2),
                                cy + big_radius * np.sin(a2), z2], dtype=float)
                free_configs.append({
                    'id': self.num_route + i,
                    'starting': start,
                    'destination': end,
                    'waypoints': [start, end],
                    'n_points': 2,
                    'priority': 2,
                    'safety_radius': 2.0,
                    'type': 'free',
                })

            if self._validate_starting_positions(route_configs, free_configs, MIN_START_DIST):
                return route_configs, free_configs

        print(f"⚠️  V3警告: 经过 {MAX_RETRIES} 次尝试仍未找到满足间距的场景，使用最后一次结果。")
        return route_configs, free_configs
