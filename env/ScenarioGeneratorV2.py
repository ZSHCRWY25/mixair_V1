"""
🆕 新场景生成器 V2 - 改进的同心圆场景
路线模式：从大圆进入 → 中圆 → 小圆 → 大圆飞出

这个版本保持与原项目的兼容性，同时提供了更合理的航路设计
"""

import numpy as np
from env.RouteGenerator import RouteGenerator


class ScenarioGeneratorV2:
    """
    改进的场景生成器 - 提供更灵活的路线设计
    
    特点：
    1. 路线从大圆一侧进入，依次经过中圆、小圆，从大圆另一侧飞出
    2. 保持与原 ScenarioGenerator 相同的输出格式
    3. 支持自定义圆的配置
    """
    
    def __init__(self, num_route, num_free, route_generator=None,
                 max_circle_diameter=100, circle_diameters=None, scene_seed=None,
                 use_random_scene=False,
                 xlarge_diameter=None,
                 route_z_min=24, route_z_max=28,
                 free_z_min=20, free_z_max=22,
                 center=None,
                 **kwargs):
        """
        初始化 V2 场景生成器

        Args:
            num_route: 航路无人机数量
            num_free: 自由无人机数量
            route_generator: 预设路线生成器（用于备用）
            max_circle_diameter: 大圆直径（自由UAV活动圈）
            circle_diameters: 各圆的直径列表 [大, 中, 小]
            scene_seed: 随机种子
            xlarge_diameter: 超大圆直径（航路UAV起终点），默认为大圆的1.3倍
            route_z_min/max: 航路UAV高度范围（默认 24-28m）
            free_z_min/max: 自由UAV高度范围（默认 20-22m）
            center: 场景中心 [x, y]，默认 (100, 100) 使所有坐标为正
        """
        self.num_route = num_route
        self.num_free = num_free
        self.route_generator = route_generator or RouteGenerator()
        self.max_circle_diameter = max_circle_diameter
        self.scene_seed = scene_seed

        # 设置圆的直径：[大, 中, 小]
        if circle_diameters is None:
            self.circle_diameters = [
                max_circle_diameter,
                max_circle_diameter * 0.7,
                max_circle_diameter * 0.4
            ]
        else:
            self.circle_diameters = circle_diameters

        # 超大圆：航路UAV起终点，默认为大圆的1.3倍
        self.xlarge_diameter = xlarge_diameter if xlarge_diameter is not None else max_circle_diameter * 1.3

        # 各类UAV高度范围（分离以减少起点冲突）
        self.route_z_min = route_z_min
        self.route_z_max = route_z_max
        self.free_z_min = free_z_min
        self.free_z_max = free_z_max

        # 场景中心偏移：使所有坐标为正（xlarge_r=65m, center=100 → 坐标范围 [35,165]）
        cx, cy = (center[0], center[1]) if center is not None else (100.0, 100.0)
        self.center = np.array([cx, cy, 0.0], dtype=float)

    @property
    def scene_radius(self):
        """场景最大半径（用于渲染边界计算）"""
        return self.xlarge_diameter / 2.0

    def _sample_point_on_circle(self, diameter, angle=None, z=None, z_min=None, z_max=None):
        """在圆周上采样一个点"""
        radius = diameter / 2.0
        if angle is None:
            angle = np.random.uniform(0, 2 * np.pi)
        if z is None:
            lo = z_min if z_min is not None else self.free_z_min
            hi = z_max if z_max is not None else self.free_z_max
            z = np.random.uniform(lo, hi)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x, y, z], dtype=float), angle
    
    def _generate_spiral_route(self, start_angle_offset=None):
        """
        生成随机螺旋路线（8航点版本）

        路径：超大圆(入) → 大圆 → 中圆 → 小圆 → 小圆 → 中圆 → 大圆 → 超大圆(出)

        航路UAV从超大圆出发，在进入自由UAV活动区（大圆）前有充足的缓冲距离，
        供RL策略观察并做出避撞决策。

        Args:
            start_angle_offset: 起始角度偏移（可选，若为None则完全随机）

        Returns:
            waypoints: 航点列表（共8个航点）
            metadata: 包含此次生成的参数（用于调试）
        """
        waypoints = []

        # 获取各圆半径
        xlarge_r = self.xlarge_diameter / 2.0
        big_radius = self.circle_diameters[0] / 2.0
        mid_radius = self.circle_diameters[1] / 2.0
        small_radius = self.circle_diameters[2] / 2.0

        # ==================== 随机化参数 ====================

        # 1. 转向角（120-160度）
        angle_span_deg = np.random.uniform(120, 160)
        angle_span = np.deg2rad(angle_span_deg)

        # 2. 起始角度
        if start_angle_offset is None:
            angle1 = np.random.uniform(0, 2 * np.pi)
        else:
            angle1 = start_angle_offset
        angle2 = angle1 + angle_span

        # 3. 高度（使用航路UAV专属范围）
        z_lo, z_hi = self.route_z_min, self.route_z_max
        z_pattern = np.random.randint(0, 3)
        if z_pattern == 0:
            # 稳定高度模式
            base_z = np.random.uniform(z_lo, z_hi)
            z_vals = [np.clip(base_z + np.random.uniform(-1, 1), z_lo, z_hi) for _ in range(8)]
        elif z_pattern == 1:
            # 升降模式（外高内低）
            start_z = np.random.uniform(z_lo, z_hi)
            end_z = np.random.uniform(z_lo, z_hi)
            fracs = [0.0, 0.1, 0.3, 0.5, 0.5, 0.7, 0.9, 1.0]
            z_vals = [start_z + (end_z - start_z) * f for f in fracs]
        else:
            # 自由随机模式
            z_vals = [np.random.uniform(z_lo, z_hi) for _ in range(8)]

        # 4. 大/中/小圆轻微缩放（超大圆不缩放）
        max_allowed_radius = self.max_circle_diameter / 2.0
        max_scale = max_allowed_radius / big_radius
        scale_factor = np.random.uniform(0.95, min(1.05, max_scale))
        big_r = big_radius * scale_factor
        mid_r = mid_radius * scale_factor
        small_r = small_radius * scale_factor

        # ==================== 生成8个航点 ====================

        # p0 【起点】超大圆 - 进入点
        waypoints.append(np.array([xlarge_r * np.cos(angle1), xlarge_r * np.sin(angle1), z_vals[0]], dtype=float))
        # p1 【过渡】大圆 - 第一组
        waypoints.append(np.array([big_r * np.cos(angle1),    big_r * np.sin(angle1),    z_vals[1]], dtype=float))
        # p2 【中间】中圆 - 第一组
        waypoints.append(np.array([mid_r * np.cos(angle1),    mid_r * np.sin(angle1),    z_vals[2]], dtype=float))
        # p3 【最深】小圆 - 第一组
        waypoints.append(np.array([small_r * np.cos(angle1),  small_r * np.sin(angle1),  z_vals[3]], dtype=float))
        # p4 【转向】小圆 - 第二组
        waypoints.append(np.array([small_r * np.cos(angle2),  small_r * np.sin(angle2),  z_vals[4]], dtype=float))
        # p5 【中间】中圆 - 第二组
        waypoints.append(np.array([mid_r * np.cos(angle2),    mid_r * np.sin(angle2),    z_vals[5]], dtype=float))
        # p6 【过渡】大圆 - 第二组
        waypoints.append(np.array([big_r * np.cos(angle2),    big_r * np.sin(angle2),    z_vals[6]], dtype=float))
        # p7 【终点】超大圆 - 飞出点
        waypoints.append(np.array([xlarge_r * np.cos(angle2), xlarge_r * np.sin(angle2), z_vals[7]], dtype=float))

        # 应用场景中心偏移（x, y 平移；z 不变）
        offset = self.center
        waypoints = [wp + np.array([offset[0], offset[1], 0.0]) for wp in waypoints]

        metadata = {
            'start_angle': angle1,
            'angle_span': angle_span_deg,
            'z_pattern': z_pattern,
            'scale_factor': scale_factor
        }

        return waypoints, metadata
    
    def _validate_starting_positions(self, route_configs, free_configs, min_dist=5.0):
        """检查所有起始位置两两间距均 >= min_dist，满足返回 True。"""
        starts = [cfg['starting'] for cfg in route_configs] + \
                 [cfg['starting'] for cfg in free_configs]
        for i in range(len(starts)):
            for j in range(i + 1, len(starts)):
                if np.linalg.norm(starts[i] - starts[j]) < min_dist:
                    return False
        return True

    def generate_random_scene_v2(self):
        """
        生成改进的随机场景（每个航路都是独立随机的）

        路线设计（改进版本）：
        - 每个航路 UAV 有独立的随机航路（不是固定模式）
        - 从大圆进入 → 中圆 → 小圆 → 小圆(另一边) → 中圆(另一边) → 大圆飞出
        - 转向夹角随机在 120-160° 范围（满足运动学约束）
        - 每条路线的高度模式、圆形缩放都不同
        - 自由 UAV 在大圆边界上随机活动

        场景复杂性：每次生成的航路都不同，增加训练的多样性
        保证：所有起始位置两两间距 >= 2.5m（> 碰撞半径之和 2.4m），
        避免无人机在起点就发生碰撞。
        """
        # 碰撞半径之和为 2.4m + 2.6m 保护区
        MIN_START_DIST = 5.0
        MAX_RETRIES = 300

        max_diameter = self.circle_diameters[0]
        max_radius = max_diameter / 2.0

        for _ in range(MAX_RETRIES):
            route_configs = []

            # 为每个航路 UAV 生成独立的随机螺旋路线
            for i in range(self.num_route):
                waypoints, metadata = self._generate_spiral_route(start_angle_offset=None)
                cfg = {
                    'id': i,
                    'starting': waypoints[0],
                    'destination': waypoints[-1],
                    'waypoints': waypoints,
                    'n_points': len(waypoints),
                    'priority': 1,
                    'type': 'spiral_randomized',
                    'start_angle': metadata['start_angle'],
                    'turning_angle': metadata['angle_span'],
                    'z_pattern': metadata['z_pattern'],
                    'scale_factor': metadata['scale_factor']
                }
                route_configs.append(cfg)

            # 自由 UAV 在大圆边界上随机（使用自由UAV专属高度范围）
            cx, cy = self.center[0], self.center[1]
            free_configs = []
            for i in range(self.num_free):
                angle1 = np.random.uniform(0, 2 * np.pi)
                z1 = np.random.uniform(self.free_z_min, self.free_z_max)
                start = np.array([cx + max_radius * np.cos(angle1),
                                  cy + max_radius * np.sin(angle1), z1], dtype=float)

                angle2 = angle1 + np.random.uniform(np.pi - np.pi / 4, np.pi + np.pi / 4)
                z2 = np.random.uniform(self.free_z_min, self.free_z_max)
                end = np.array([cx + max_radius * np.cos(angle2),
                                cy + max_radius * np.sin(angle2), z2], dtype=float)

                cfg = {
                    'id': self.num_route + i,
                    'starting': start,
                    'destination': end,
                    'waypoints': [start, end],
                    'n_points': 2,
                    'priority': 2,
                    'safety_radius': 2.0,
                    'type': 'free'
                }
                free_configs.append(cfg)

            if self._validate_starting_positions(route_configs, free_configs, MIN_START_DIST):
                return route_configs, free_configs

        # 超出重试次数，使用最后一次生成结果并打印警告
        print(f"⚠️  警告: 经过 {MAX_RETRIES} 次尝试仍未找到起点间距均 >= {MIN_START_DIST}m 的场景，"
              f"使用最后一次结果。")
        return route_configs, free_configs
    
    def generate(self):
        """
        生成场景（始终使用V2版本的改进路线）
        
        Returns:
            route_configs: 航路配置列表
            free_configs: 自由UAV配置列表
        """
        return self.generate_random_scene_v2()


# ============================================================================
# 兼容性函数 - 让 ScenarioGeneratorV2 能直接替换原版本
# ============================================================================

def compare_scenario_versions(num_tests=10):
    """
    比较两个版本的场景生成
    用于性能测试和结果对比
    """
    import time
    from env.ScenarioGenerator import ScenarioGenerator
    
    print("\n" + "="*70)
    print("🔄 场景生成器版本对比")
    print("="*70)
    
    # V1（原版本）
    gen_v1 = ScenarioGenerator(5, 10, use_random_scene=True, max_circle_diameter=100)
    times_v1 = []
    for _ in range(num_tests):
        start = time.perf_counter()
        gen_v1.generate()
        times_v1.append((time.perf_counter() - start) * 1000)
    
    # V2（新版本）
    gen_v2 = ScenarioGeneratorV2(5, 10, max_circle_diameter=100)
    times_v2 = []
    for _ in range(num_tests):
        start = time.perf_counter()
        gen_v2.generate()
        times_v2.append((time.perf_counter() - start) * 1000)
    
    print(f"\n📊 性能对比:")
    print(f"  V1 (原版本) 平均耗时: {np.mean(times_v1):.3f} ms")
    print(f"  V2 (新版本) 平均耗时: {np.mean(times_v2):.3f} ms")
    
    # 生成一个场景用于结构对比
    route_v1, free_v1 = gen_v1.generate()
    route_v2, free_v2 = gen_v2.generate()
    
    print(f"\n📋 结构对比:")
    print(f"  V1 航路配置: {len(route_v1)} 个")
    print(f"    航路0航点数: {route_v1[0]['n_points']}")
    print(f"  V2 航路配置: {len(route_v2)} 个")
    print(f"    航路0航点数: {route_v2[0]['n_points']}")
    
    print(f"\n✨ V2 新特性:")
    print(f"  ✓ 改进的螺旋路线设计")
    print(f"  ✓ 路线类型标记 ('spiral' vs 'free')")
    print(f"  ✓ 记录起始角度便于调试")
    
    return gen_v1, gen_v2


if __name__ == '__main__':
    # 简单测试
    print("测试 ScenarioGeneratorV2...")
    gen = ScenarioGeneratorV2(5, 10, max_circle_diameter=100)
    routes, frees = gen.generate()
    
    print(f"✓ 生成成功: {len(routes)} 个航路, {len(frees)} 个自由UAV")
    print(f"\n路由0详情:")
    print(f"  起点: {routes[0]['starting'].round(2)}")
    print(f"  终点: {routes[0]['destination'].round(2)}")
    print(f"  航点数: {routes[0]['n_points']} (改进版本为 6 个)")
    print(f"  类型: {routes[0]['type']}")
    print(f"  起始角度: {np.rad2deg(routes[0]['start_angle']):.1f}°")
    print(f"  转向夹角: {routes[0]['turning_angle']}°")
    
    print(f"\n🔄 路线结构（符合运动学原理）:")
    print(f"  1️⃣  大圆(进入) → 2️⃣  中圆(v1) → 3️⃣  小圆(v1)")
    print(f"  ↓")
    print(f"  4️⃣  小圆(v2) → 5️⃣  中圆(v2) → 6️⃣  大圆(飞出)")
    print(f"\n✅ 运动学约束: 转向夹角 {routes[0]['turning_angle']}° > 120° ✓")
