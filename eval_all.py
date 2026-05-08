#!/usr/bin/env python
"""
批量实验评估脚本
对 experiments/ 下每个已完成的任务，在不同自由UAV数量下评估避撞成功率。
使用确定性动作（dist.mean），不采样，结果保存到 experiments/eval_summary.csv。
"""

import os
import sys
import csv
import glob
import yaml
import warnings

# 必须在导入 matplotlib 之前设置后端
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch

from env.uavenv import UAVEnv
from config import Config


# ============================================================================
# 评估参数
# ============================================================================

FREE_UAV_COUNTS = [0, 2, 4, 6, 8, 10]  # 评估时依次测试的自由UAV数量
EVAL_EPISODES = 20                       # 每个组合的评估 episode 数
EVAL_SEED = 42                           # 固定随机种子
MAX_EP_LEN = 500                         # 每个 episode 最大步数


def set_seed(seed: int):
    """固定所有随机种子以保证可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(model_dir: str):
    """在 model_dir 中查找 epoch 号最大的 .pt 文件"""
    candidates = glob.glob(os.path.join(model_dir, 'policy_*.pt'))
    if not candidates:
        # 尝试子目录（兼容 get_model_save_dir 产生的嵌套结构）
        for root, _dirs, files in os.walk(model_dir):
            for f in files:
                if f.startswith('policy_') and f.endswith('.pt'):
                    candidates.append(os.path.join(root, f))
    if not candidates:
        return None

    def _epoch_num(p: str):
        basename = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(basename.split('epoch')[-1])
        except (ValueError, IndexError):
            return 0

    candidates.sort(key=_epoch_num)
    return candidates[-1]


def load_policy(ckpt_path: str, device: str):
    """从 checkpoint 加载 TransformerPolicyV2"""
    from policy.transformer_policy_v2 import TransformerPolicyV2

    policy = TransformerPolicyV2(
        self_state_dim=Config.SELF_STATE_DIM,
        other_route_dim=Config.OTHER_ROUTE_DIM,
        free_uav_dim=Config.FREE_UAV_DIM,
        max_free_uavs=Config.MAX_FREE_UAVS,
        embed_dim=Config.EMBED_DIM,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        mlp_hidden=Config.MLP_HIDDEN,
        device=device,
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()
    return policy


def run_episode(env, policy, num_route_uavs, device):
    """运行一个 episode，返回 (success, collisions, conflicts)"""
    obs_list, _ = env.reset(regenerate_scene=True)
    done = False

    while not done:
        actions = []
        with torch.no_grad():
            for i in range(num_route_uavs):
                obs_t = torch.as_tensor(obs_list[i], dtype=torch.float32, device=device).unsqueeze(0)
                dist = policy.get_dist(obs_t)
                # 确定性动作：使用分布均值
                action = dist.mean.item()
                actions.append(action)

        obs_list, _rewards, terminated, truncated, info = env.step(np.array(actions))
        done = terminated or truncated

    route_route = info.get('route_route_collisions', 0)
    route_free = info.get('route_free_collisions', 0)
    total_collisions = route_route + route_free
    out_of_bounds = info.get('out_of_bounds', False)
    conflicts = info.get('conflict_count', 0)

    success = (total_collisions == 0 and not out_of_bounds)
    return success, total_collisions, conflicts


def evaluate_task(task_dir: str, task_name: str):
    """
    评估单个已完成任务，遍历所有 free_uav 数量。
    返回 [{task_name, free_uav, success_rate, avg_collisions, avg_conflicts, std_success}, ...]
    """
    # 读取配置快照
    snapshot_path = os.path.join(task_dir, 'config_snapshot.yaml')
    if os.path.exists(snapshot_path):
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            snap = yaml.safe_load(f) or {}
    else:
        snap = {}

    scenario_version = snap.get('SCENARIO_VERSION', 'v3')
    num_route_uavs = snap.get('NUM_ROUTE_UAVS', 5)
    max_circle_diameter = snap.get('MAX_CIRCLE_DIAMETER', 100)
    circle_diameters = snap.get('CIRCLE_DIAMETERS', None)
    max_free_uavs = snap.get('MAX_FREE_UAVS', 5)

    # 应用场景基础配置
    Config.update({
        'SCENARIO_VERSION': scenario_version,
        'NUM_ROUTE_UAVS': num_route_uavs,
        'MAX_CIRCLE_DIAMETER': max_circle_diameter,
        'CIRCLE_DIAMETERS': circle_diameters,
        'MAX_FREE_UAVS': max_free_uavs,
    })

    # 查找模型
    model_dir = os.path.join(task_dir, 'models')
    ckpt_path = find_latest_checkpoint(model_dir)
    if ckpt_path is None:
        print(f"  [WARN] {task_name}: 未找到模型 checkpoint，跳过")
        return []

    print(f"  模型: {ckpt_path}")
    device = Config.DEVICE
    policy = load_policy(ckpt_path, device)

    results = []

    for n_free in FREE_UAV_COUNTS:
        # 创建环境（临时修改 NUM_FREE_UAVS）
        env = UAVEnv(
            num_route_uavs=num_route_uavs,
            num_free_uavs=n_free,
            max_steps=MAX_EP_LEN,
            render_mode=None,
            use_random_scene=True,
            max_circle_diameter=max_circle_diameter,
            circle_diameters=circle_diameters,
        )

        successes = []
        all_collisions = []
        all_conflicts = []

        for ep in range(EVAL_EPISODES):
            set_seed(EVAL_SEED + ep)  # 每个 episode 使用不同种子以覆盖场景多样性
            success, collisions, conflicts = run_episode(env, policy, num_route_uavs, device)
            successes.append(success)
            all_collisions.append(collisions)
            all_conflicts.append(conflicts)

        success_rate = float(np.mean(successes))
        success_std = float(np.std(successes))
        avg_collisions = float(np.mean(all_collisions))
        avg_conflicts = float(np.mean(all_conflicts))

        results.append({
            'task_name': task_name,
            'free_uav': n_free,
            'success_rate': round(success_rate, 4),
            'std_success': round(success_std, 4),
            'avg_collisions': round(avg_collisions, 2),
            'avg_conflicts': round(avg_conflicts, 2),
        })

        print(f"  free_uav={n_free:2d}  success={success_rate:.2%}±{success_std:.2%}"
              f"  collisions={avg_collisions:.2f}  conflicts={avg_conflicts:.2f}")

        env.close()

    return results


def main():
    experiments_root = './experiments'
    if not os.path.isdir(experiments_root):
        print(f"错误：未找到 {experiments_root}/ 目录，请先运行 train_all.py")
        return

    # 扫描已完成任务
    tasks = []
    for entry in sorted(os.listdir(experiments_root)):
        task_dir = os.path.join(experiments_root, entry)
        if not os.path.isdir(task_dir):
            continue
        flag_file = os.path.join(task_dir, 'completed.flag')
        if os.path.exists(flag_file):
            tasks.append((entry, task_dir))

    if not tasks:
        print("没有找到已完成的任务（缺少 completed.flag），请先运行 train_all.py")
        return

    print("=" * 60)
    print(f"批量评估 — 共 {len(tasks)} 个任务")
    print(f"自由UAV测试取值: {FREE_UAV_COUNTS}")
    print(f"每个组合评估: {EVAL_EPISODES} episodes, seed={EVAL_SEED}")
    print("=" * 60)

    all_results = []

    for task_name, task_dir in tasks:
        print(f"\n[EVAL] {task_name}")
        results = evaluate_task(task_dir, task_name)
        all_results.extend(results)

    # 保存汇总结果
    output_path = os.path.join(experiments_root, 'eval_summary.csv')
    if all_results:
        fieldnames = ['task_name', 'free_uav', 'success_rate', 'std_success',
                      'avg_collisions', 'avg_conflicts']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n评估结果已保存到: {output_path}")
    else:
        print("\n没有产生任何评估结果")

    print("=" * 60)
    print("评估完成")
    print("=" * 60)


if __name__ == '__main__':
    # 抑制 gymnasium 的警告
    warnings.filterwarnings('ignore')
    main()
