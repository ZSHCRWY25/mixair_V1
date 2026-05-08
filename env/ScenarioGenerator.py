import numpy as np
from env.RouteGenerator import RouteGenerator

class ScenarioGenerator:
    def __init__(self, num_route, num_free, route_generator=None, use_random_scene=False, 
                 max_circle_diameter=100, circle_diameters=None, scene_seed=None):
        """
        场景生成器
        
        Args:
            num_route: 航路无人机数量
            num_free: 自由无人机数量
            route_generator: 预设路线生成器（用于固定场景模式）
            use_random_scene: 是否使用随机场景（同心圆）
            max_circle_diameter: 最大圆直径（同心圆模式）
            circle_diameters: 各圆的直径列表，如 [100, 70, 40]，若为 None 则自动生成
            scene_seed: 随机种子（用于复现）
        """
        self.num_route = num_route
        self.num_free = num_free
        self.route_generator = route_generator or RouteGenerator()
        self.use_random_scene = use_random_scene
        self.max_circle_diameter = max_circle_diameter
        self.scene_seed = scene_seed
        
        # 设置圆的直径
        if circle_diameters is None:
            # 自动生成：最大圆、中圆、小圆
            self.circle_diameters = [
                max_circle_diameter,
                max_circle_diameter * 0.7,
                max_circle_diameter * 0.4
            ]
        else:
            self.circle_diameters = circle_diameters

    def _sample_on_circle(self, diameter):
        """在圆周上随机选择一个点"""
        radius = diameter / 2.0
        angle = np.random.uniform(0, 2 * np.pi)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = np.random.uniform(50, 150)  # 高度随机
        return np.array([x, y, z], dtype=float)

    def _sample_opposite_points_on_circle(self, diameter):
        """在圆周上随机选择对向的两个点"""
        radius = diameter / 2.0
        angle1 = np.random.uniform(0, 2 * np.pi)
        angle2 = angle1 + np.pi  # 对面
        
        point1 = np.array([
            radius * np.cos(angle1),
            radius * np.sin(angle1),
            np.random.uniform(50, 150)
        ], dtype=float)
        
        point2 = np.array([
            radius * np.cos(angle2),
            radius * np.sin(angle2),
            np.random.uniform(50, 150)
        ], dtype=float)
        
        return point1, point2

    def generate_random_scene(self):
        """
        生成随机场景（同心圆配置）
        - 航路 UAV：在三个同心圆上各选一对对向点，共 6 个航点
        - 自由 UAV：只在最大圆上选一对对向点
        """
        route_configs = []
        
        # 为每个航路 UAV 生成 6 个航点（3 个圆，每个圆 2 个点）
        for i in range(self.num_route):
            waypoints = []
            # 在三个同心圆上选点
            for circle_diameter in self.circle_diameters:
                p1, p2 = self._sample_opposite_points_on_circle(circle_diameter)
                waypoints.extend([p1, p2])
            
            cfg = {
                'id': i,
                'starting': waypoints[0],
                'destination': waypoints[-1],
                'waypoints': waypoints,
                'n_points': len(waypoints),
                'priority': 1,
            }
            route_configs.append(cfg)
        
        # 为每个自由 UAV 生成配置（只在最大圆上）
        free_configs = []
        max_diameter = self.circle_diameters[0]
        for i in range(self.num_free):
            start, end = self._sample_opposite_points_on_circle(max_diameter)
            cfg = {
                'id': self.num_route + i,
                'starting': start,
                'destination': end,
                'waypoints': [start, end],
                'n_points': 2,
                'priority': 2,
                'safety_radius': 2.0
            }
            free_configs.append(cfg)
        
        return route_configs, free_configs

    def generate(self):
        """生成场景（自动选择固定或随机）"""
        if self.use_random_scene:
            return self.generate_random_scene()
        else:
            # 原有的固定场景生成逻辑
            route_configs = []
            for i in range(self.num_route):
                waypoints = self.route_generator.generate_route()
                cfg = {
                    'id': i,
                    'starting': waypoints[0],
                    'destination': waypoints[-1],
                    'waypoints': waypoints,
                    'n_points': len(waypoints),
                    'priority': 1,
                }
                route_configs.append(cfg)

            free_configs = []
            for i in range(self.num_free):
                start = np.random.uniform(-100, 100, size=3)
                end = np.random.uniform(-100, 100, size=3)
                cfg = {
                    'id': self.num_route + i,
                    'starting': start,
                    'destination': end,
                    'waypoints': [start, end],
                    'n_points': 2,
                    'priority': 2,
                    'safety_radius': 2.0
                }
                free_configs.append(cfg)

            return route_configs, free_configs
    



    # test_scenario_generator.py
import matplotlib.pyplot as plt

def test_scenario_generator():
    # 创建场景生成器：3架航路无人机，5架自由无人机
    generator = ScenarioGenerator(num_route=3, num_free=5)
    route_configs, free_configs = generator.generate()

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.grid(True)
    ax.set_title('UAV Scenario (Routes, Start & Goal)')

    # 1. 绘制航路无人机的航路（蓝色虚线）
    for cfg in route_configs:
        waypoints = np.array(cfg['waypoints'])
        ax.plot(waypoints[:, 0], waypoints[:, 1], 'b--', alpha=0.6, linewidth=1.5,
                label='Route UAV Path' if cfg['id'] == 0 else "")

    # 2. 绘制自由无人机的直线航路（灰色虚线）
    for cfg in free_configs:
        waypoints = np.array(cfg['waypoints'])
        ax.plot(waypoints[:, 0], waypoints[:, 1], 'gray', linestyle=':', alpha=0.5,
                label='Free UAV Path' if cfg['id'] == 2 else "")

    # 3. 绘制起点（绿色圆点）和终点（红色星号）
    for cfg in route_configs + free_configs:
        ax.plot(cfg['starting'][0], cfg['starting'][1], 'go', markersize=8,
                label='Start' if cfg['id'] == 0 else "")
        ax.plot(cfg['destination'][0], cfg['destination'][1], 'r*', markersize=10,
                label='Goal' if cfg['id'] == 0 else "")

    # 为每个无人机添加文本标签（可选）
    for cfg in route_configs + free_configs:
        ax.text(cfg['starting'][0], cfg['starting'][1], f" {cfg['id']}", fontsize=8, verticalalignment='center')
        ax.text(cfg['destination'][0], cfg['destination'][1], f" {cfg['id']}", fontsize=8, verticalalignment='center')

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_scenario_generator()