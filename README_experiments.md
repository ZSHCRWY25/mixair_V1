# 批量实验使用指南

## 目录结构

```
experiments/
├── v2_route5/                  # 任务 1：V2 场景，航路UAV=5
│   ├── models/                 # 模型 checkpoint（由训练自动保存）
│   │   └── policy_transformer_epoch300.pt
│   ├── config_snapshot.yaml    # 训练时的配置快照
│   ├── completed.flag          # 完成标志（训练完成后生成）
│   └── training_metrics.xlsx   # 训练指标（由训练自动保存）
├── v3_route5/                  # 任务 2：V3 场景，航路UAV=5
│   └── ...
└── eval_summary.csv            # 评估汇总结果
```

## 运行方式

### 1. 批量训练

```bash
python train_all.py
```

- 按 `TASK_LIST` 顺序执行每个实验任务
- 已存在 `completed.flag` 的任务自动跳过（支持断点续跑）
- 训练失败的任务不生成 flag，下次运行时会自动重试

### 2. 批量评估

```bash
python eval_all.py
```

- 自动扫描 `experiments/` 下所有含 `completed.flag` 的任务
- 对每个任务分别测试 free_uav = 0, 2, 4, 6, 8, 10
- 每个 (task, free_uav) 组合运行 20 个 episode（seed=42）
- 使用确定性策略（dist.mean，不采样）
- 结果保存到 `experiments/eval_summary.csv`

### 3. 添加新任务

编辑 `train_all.py` 中的 `TASK_LIST`，按格式添加：

```python
{
    'task_name': 'v3_route8',
    'scenario_version': 'v3',
    'num_route_uavs': 8,
    'num_free_uavs': 0,
    'train_epochs': 300,
},
```

## 评估指标说明

`eval_summary.csv` 包含以下列：

| 列名 | 含义 |
|------|------|
| task_name | 任务名称 |
| free_uav | 测试时的自由UAV数量 |
| success_rate | 成功率（无碰撞且无出界） |
| std_success | 成功率标准差 |
| avg_collisions | 平均碰撞次数 |
| avg_conflicts | 平均冲突次数（TTC<3s） |

## 注意事项

- 训练脚本默认使用 Transformer 策略 + 随机同心圆场景 + 课程学习
- 评估脚本使用固定种子（seed=42），每个 episode 递增种子以保证场景多样性
- 评估时关闭渲染以加速，matplotlib 后端设为 Agg
- 原有 `models/` 文件夹不受批量实验影响（实验产物完全隔离到 `experiments/`）
