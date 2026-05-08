#!/usr/bin/env python
"""
批量实验训练脚本
按顺序执行多个实验任务，每个任务隔离保存到 experiments/{task_name}/
支持断点续跑（通过 completed.flag 跳过已完成任务）
"""

import os
import sys
import glob
import yaml
import traceback

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import Config

# ============================================================================
# 任务定义
# ============================================================================

TASK_LIST = [
    # ========================================================================
    # 基础训练任务：航路UAV=6，自由UAV=0，训练500轮
    # ========================================================================
    {
        'task_name': 'v2_route6',
        'scenario_version': 'v2',
        'num_route_uavs': 6,
        'num_free_uavs': 0,
        'train_epochs': 500,
    },
    {
        'task_name': 'v3_route6',
        'scenario_version': 'v3',
        'num_route_uavs': 6,
        'num_free_uavs': 0,
        'train_epochs': 500,
    },

    # ========================================================================
    # 微调任务：基于基础模型，加入自由UAV=1,2,3 继续训练200轮
    # ========================================================================
    {
        'task_name': 'v2_route6_finetune_free1',
        'scenario_version': 'v2',
        'num_route_uavs': 6,
        'num_free_uavs': 1,
        'train_epochs': 200,
        'resume_from_task': 'v2_route6',
    },
    {
        'task_name': 'v2_route6_finetune_free2',
        'scenario_version': 'v2',
        'num_route_uavs': 6,
        'num_free_uavs': 2,
        'train_epochs': 200,
        'resume_from_task': 'v2_route6',
    },
    {
        'task_name': 'v2_route6_finetune_free3',
        'scenario_version': 'v2',
        'num_route_uavs': 6,
        'num_free_uavs': 3,
        'train_epochs': 200,
        'resume_from_task': 'v2_route6',
    },
    {
        'task_name': 'v3_route6_finetune_free1',
        'scenario_version': 'v3',
        'num_route_uavs': 6,
        'num_free_uavs': 1,
        'train_epochs': 200,
        'resume_from_task': 'v3_route6',
    },
    {
        'task_name': 'v3_route6_finetune_free2',
        'scenario_version': 'v3',
        'num_route_uavs': 6,
        'num_free_uavs': 2,
        'train_epochs': 200,
        'resume_from_task': 'v3_route6',
    },
    {
        'task_name': 'v3_route6_finetune_free3',
        'scenario_version': 'v3',
        'num_route_uavs': 6,
        'num_free_uavs': 3,
        'train_epochs': 200,
        'resume_from_task': 'v3_route6',
    },
]


def run_task(task):
    """执行单个实验任务"""
    task_name = task['task_name']
    exp_dir = os.path.join('./experiments', task_name)
    flag_file = os.path.join(exp_dir, 'completed.flag')
    model_dir = os.path.join(exp_dir, 'models')

    # 断点续跑：已完成的任务跳过
    if os.path.exists(flag_file):
        print(f"\n{'='*60}")
        print(f"[SKIP] {task_name} — completed.flag 已存在，跳过训练")
        print(f"{'='*60}")
        return

    os.makedirs(model_dir, exist_ok=True)

    # ── 微调模式：检查基础任务并定位 checkpoint ─────────────────
    resume_from_task = task.get('resume_from_task')
    is_finetune = resume_from_task is not None

    if is_finetune:
        base_dir = os.path.join('./experiments', resume_from_task)
        base_flag = os.path.join(base_dir, 'completed.flag')
        if not os.path.exists(base_flag):
            print(f"  [WARN] 基础任务 {resume_from_task} 未完成（缺少 completed.flag），跳过微调")
            return
        # 在基础任务模型目录中查找 epoch 最大的 checkpoint
        base_model_dir = os.path.join(base_dir, 'models')
        candidates = glob.glob(os.path.join(base_model_dir, '**', 'policy_*.pt'), recursive=True)
        if not candidates:
            print(f"  [WARN] 基础任务 {resume_from_task} 模型目录中未找到 .pt 文件，跳过微调")
            return

        def _epoch_num(p):
            basename = os.path.splitext(os.path.basename(p))[0]
            try:
                return int(basename.split('epoch')[-1])
            except (ValueError, IndexError):
                return 0
        candidates.sort(key=_epoch_num)
        ckpt_path = candidates[-1]
        Config.RESUME_FROM = ckpt_path
    else:
        Config.RESUME_FROM = None

    # ── 训练状态标签 ────────────────────────────────────────────
    mode_label = "[FINE-TUNE]" if is_finetune else "[TRAIN]"
    print(f"\n{'='*60}")
    print(f"{mode_label} {task_name}")
    print(f"  场景版本: {task['scenario_version']}")
    print(f"  航路UAV: {task['num_route_uavs']}, 自由UAV: {task['num_free_uavs']}")
    print(f"  训练轮数: {task['train_epochs']}")
    print(f"  输出目录: {exp_dir}")
    if is_finetune:
        print(f"  基础任务: {resume_from_task}")
        print(f"  加载 checkpoint: {Config.RESUME_FROM}")
    print(f"{'='*60}")

    try:
        # ── 1. 应用任务配置 ─────────────────────────────────────
        Config.update({
            'SCENARIO_VERSION': task['scenario_version'],
            'NUM_ROUTE_UAVS': task['num_route_uavs'],
            'NUM_FREE_UAVS': task['num_free_uavs'],
            'TRAIN_EPOCHS': task['train_epochs'],
            'USE_RANDOM_SCENE': True,
            'POLICY_VERSION': 'transformer',
            'RENDER_MODE': None,
            'RENDER_EVERY_STEP': False,
            'TRAJECTORY_MODE': False,
            'CURRICULUM_LEARNING': False if is_finetune else True,
        })
        Config.set_save_dir(model_dir)

        # ── 2. 保存配置快照 ─────────────────────────────────────
        snapshot_path = os.path.join(exp_dir, 'config_snapshot.yaml')
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            yaml.dump(Config.snapshot(), f, default_flow_style=False, allow_unicode=True)

        # ── 3. 启动训练 ─────────────────────────────────────────
        import train_complete
        train_complete.main()

        # ── 4. 标记完成 ─────────────────────────────────────────
        with open(flag_file, 'w', encoding='utf-8') as f:
            f.write(f"completed at {__import__('datetime').datetime.now()}\n")
        print(f"\n[OK] {task_name} 训练完成，标志文件: {flag_file}")

    except Exception:
        print(f"\n[FAIL] {task_name} 训练异常:")
        traceback.print_exc()
    finally:
        # ── 5. 恢复默认保存目录 ─────────────────────────────────
        Config.set_save_dir(None)


def main():
    print("=" * 60)
    print("批量实验训练")
    print(f"任务总数: {len(TASK_LIST)}")
    print("=" * 60)

    for i, task in enumerate(TASK_LIST):
        print(f"\n[{i+1}/{len(TASK_LIST)}] 开始任务: {task['task_name']}")
        run_task(task)

    print("\n" + "=" * 60)
    print("所有任务执行完毕")
    print("=" * 60)


if __name__ == '__main__':
    main()
