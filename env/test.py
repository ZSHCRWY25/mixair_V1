# import time
# import numpy as np
# from uavenv import UAVEnv

# def test_env():
#     # 创建环境，启用渲染
#     env = UAVEnv(
#         num_route_uavs=3,        # 3架航路无人机
#         num_free_uavs=5,         # 5架自由无人机
#         dt=1,                  # 时间步长0.5秒
#         max_steps=1000,           # 最大步数200
#         noise=True,              # 启用控制噪声
#         control_std=[0.05, 0.05, 0.05],  # 噪声标准差
#         render_mode='human'      # 开启渲染
#     )

#     obs, info = env.reset()
#     total_reward = 0
#     step = 0
#     done = False

#     while not done:
#         # 生成随机动作（加速度），范围[-1, 1] m/s²
#         actions = np.random.uniform(-1.0, 1.0, size=env.num_route_uavs)

#         obs, reward, terminated, truncated, info = env.step(actions)
#         total_reward += reward
#         step += 1

#         print(f"Step {step:3d}: reward = {reward:6.2f} | total = {total_reward:7.2f} | done = {terminated or truncated}")

#         if terminated or truncated:
#             print("Episode finished.")
#             break

#         # 控制渲染刷新速度（避免过快）
#         time.sleep(0.05)

#     env.close()

# if __name__ == "__main__":
#     test_env()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# 假设 ScenarioGeneratorV2 在 env 模块中，若不在同一目录请调整导入路径
from ScenarioGeneratorV2 import ScenarioGeneratorV2


class Arrow3D(FancyArrowPatch):
    """用于在3D图中绘制箭头的辅助类"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def plot_scene_v2(gen, route_configs, free_configs, ax=None):
    """
    绘制V2生成的场景
    
    Args:
        gen: ScenarioGeneratorV2 实例（用于获取圆参数）
        route_configs: 航路无人机配置列表
        free_configs: 自由无人机配置列表
        ax: 可选的 matplotlib 3D axes，若为 None 则新建
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # 1. 绘制三个同心圆（参考线）
    diameters = gen.circle_diameters
    colors = ['#2E86AB', '#F18F01', '#C73E1D']  # 大、中、小
    labels = ['大圆 (进入/飞出)', '中圆', '小圆']
    theta = np.linspace(0, 2*np.pi, 100)
    z_mean = (gen.z_min + gen.z_max) / 2.0
    
    for i, (d, c, lbl) in enumerate(zip(diameters, colors, labels)):
        r = d / 2.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full_like(theta, z_mean)
        ax.plot(x, y, z, color=c, linestyle='--', linewidth=1.5, alpha=0.7, label=lbl)
        # 添加圆标注
        ax.text(r, 0, z_mean, f'{lbl}\nØ{d:.0f}', color=c, fontsize=9, ha='center')

    # 2. 绘制航路无人机路径
    route_colors = plt.cm.tab10(np.linspace(0, 1, len(route_configs)))
    
    for idx, cfg in enumerate(route_configs):
        pts = np.array(cfg['waypoints'])
        color = route_colors[idx]
        
        # 绘制连线（路径）
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
                '-o', color=color, markersize=4, linewidth=2, 
                alpha=0.9, label=f'航路 {cfg["id"]}' if idx < 3 else "")
        
        # 起点和终点特殊标记
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], 
                   color=color, marker='s', s=80, edgecolors='k', linewidths=0.8)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], 
                   color=color, marker='^', s=80, edgecolors='k', linewidths=0.8)
        
        # 添加箭头表示方向（在路径中段）
        mid_idx = len(pts) // 2
        if mid_idx < len(pts) - 1:
            arrow = Arrow3D(
                [pts[mid_idx, 0], pts[mid_idx+1, 0]],
                [pts[mid_idx, 1], pts[mid_idx+1, 1]],
                [pts[mid_idx, 2], pts[mid_idx+1, 2]],
                mutation_scale=15, lw=1.5, arrowstyle="->", color=color, alpha=0.8
            )
            ax.add_artist(arrow)
        
        # 标注航点序号
        for i, p in enumerate(pts):
            ax.text(p[0], p[1], p[2], f'{i+1}', color=color, fontsize=8, 
                    ha='center', va='center', 
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

    # 3. 绘制自由无人机
    free_color = '#6A4C93'
    for cfg in free_configs:
        start = cfg['starting']
        end = cfg['destination']
        # 起点
        ax.scatter(start[0], start[1], start[2], 
                   color=free_color, marker='o', s=50, edgecolors='k', alpha=0.7)
        # 终点
        ax.scatter(end[0], end[1], end[2], 
                   color=free_color, marker='X', s=50, edgecolors='k', alpha=0.7)
        # 连线
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                color=free_color, linestyle=':', linewidth=1, alpha=0.5)
    
    # 添加一个自由无人机示例到图例
    ax.scatter([], [], [], color=free_color, marker='o', s=30, label='自由UAV起点')
    ax.scatter([], [], [], color=free_color, marker='X', s=30, label='自由UAV终点')

    # 4. 图形装饰
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('🆕 场景生成器 V2 - 同心圆螺旋路线', fontsize=14, pad=20)
    
    # 设置等比例坐标轴
    max_range = diameters[0] * 0.6
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([gen.z_min - 5, gen.z_max + 5])
    
    # 图例（避免重复过多）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1))
    
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    
    return fig, ax


def quick_test_and_plot(num_route=4, num_free=6, seed=42):
    """
    快速生成并绘制场景
    
    Args:
        num_route: 航路无人机数量
        num_free: 自由无人机数量
        seed: 随机种子
    """
    # 设置随机种子以便复现
    np.random.seed(seed)
    
    # 创建生成器
    gen = ScenarioGeneratorV2(
        num_route=num_route,
        num_free=num_free,
        max_circle_diameter=100,
        scene_seed=seed
    )
    
    # 生成场景
    print(f"生成场景 (航路={num_route}, 自由={num_free}, seed={seed})...")
    route_configs, free_configs = gen.generate()
    
    # 打印简要信息
    print("\n📋 航路无人机信息:")
    for cfg in route_configs:
        print(f"  ID {cfg['id']}: 转向角={cfg['turning_angle']:.1f}°, "
              f"高度模式={cfg['z_pattern']}, 缩放={cfg['scale_factor']:.2f}")
    
    # 绘图
    fig, ax = plot_scene_v2(gen, route_configs, free_configs)
    plt.show()
    
    return gen, route_configs, free_configs, fig


if __name__ == '__main__':
    # 运行测试
    gen, routes, frees, fig = quick_test_and_plot(num_route=3, num_free=5, seed=123)