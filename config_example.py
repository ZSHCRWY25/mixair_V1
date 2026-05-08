"""
多智能体 PPO 训练配置 - 支持策略和场景切换

===== 策略选择 =====
VERSION = 'transformer'  # 使用 Transformer 创新版本（处理可变长度自由UAV）
VERSION = 'mlp_direct'   # 使用 MLP 直接版本（对比实验）

===== 场景选择 =====
USE_RANDOM_SCENE = True   # 使用同心圆随机场景（每轮重新生成）
USE_RANDOM_SCENE = False  # 使用固定场景
"""

import torch
import os


class Config:
    """训练配置类"""

    # [新增] 自定义保存目录（None 时使用默认 models/ 路径）
    _custom_save_dir: str | None = None

    # ==================== 策略配置 ====================
    
    # 策略版本选择
    # 可选值：'transformer' 或 'mlp_direct'
    POLICY_VERSION = 'transformer'  # 推荐：使用 Transformer 创新版本
    # POLICY_VERSION = 'mlp_direct'  # 对比：使用 MLP 直接版本
    
    # Transformer 策略参数（仅当 POLICY_VERSION='transformer' 时有效）
    SELF_STATE_DIM = 10
    OTHER_ROUTE_DIM = 16  # 8rel + ttc + prog + type_indicator + heading_align + 4 conflict_type onehot
    FREE_UAV_DIM = 10     # 8rel + ttc + path_crossing_time

    EMBED_DIM = 64
    NHEAD = 4
    NUM_LAYERS = 2
    MLP_HIDDEN = [256, 128]
    
    # ==================== 场景配置 ====================

    SCENARIO_VERSION = 'v3'

    # 场景生成模式选择
    USE_RANDOM_SCENE = True              # 使用随机同心圆场景


    # 随机场景参数（仅当 USE_RANDOM_SCENE=True 时有效）
    MAX_CIRCLE_DIAMETER = 100            # 最大圆直径（大圆直径）
    CIRCLE_DIAMETERS = None              # 如为 None，则自动生成 [100, 70, 40]

    
    # ==================== UAV 配置 ====================
    
    NUM_ROUTE_UAVS = 5                  # 航路无人机数量
    NUM_FREE_UAVS = 0                  # 自由无人机数量
    
    # ==================== 奖励配置 ====================
    
    # 碰撞惩罚（正常每步奖励约0.2-0.5，-20约为40-100步奖励，比例合理）
    COLLISION_PENALTY = -20.0
    
    # 航点奖励（【改进】动态根据航点数计算）
    WAYPOINT_REWARD_COEF = 1.0           # 航点奖励系数
    
    # 终点/全部到达奖励
    TERMINAL_REWARD_COEF = 2.0           # 终点奖励系数
    
    # 进度奖励 (【改进】增加进度奖励激励)
    PROGRESS_REWARD_SCALE = 0.5          # 进度奖励缩放系数 [0.3-1.0]
    
    # 观测归一化 (【改进】启用观测归一化)
    NORMALIZE_OBSERVATIONS = True        # 是否对观测进行归一化
    
    # ==================== 训练超参数 ====================
    
    # 训练设置
    TRAIN_EPOCHS = 300
    STEPS_PER_EPOCH = 520      # 增大：9个agent每个至少~220步
    MAX_EP_LEN = 500
    GAMMA = 0.99
    LAM = 0.95

    # PPO 参数
    CLIP_RATIO = 0.2
    TARGET_KL = 0.05
    TRAIN_PI_ITERS = 10         # 标准PPO值
    TRAIN_V_ITERS = 10          # 与pi_iters对齐，避免价值更新40次污染共享backbone
    ENTROPY_COEF = 0.01         # 熵正则化系数，鼓励探索，防止策略过早收敛
    LEARNING_RATE = 3e-4
    MAX_GRAD_NORM = 0.5
    
    # 设备
    # DEVICE = 'cpu'              # 'cpu' 或 'cuda'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 输出和检查点
    SAVE_INTERVAL = 50          # 每 N 个 epoch 保存一次
    RENDER_INTERVAL = 1         # 每 N 个 epoch 渲染一次
    SAVE_DIR = './models/'

    RESUME_FROM =  None

    # ==================== 课程学习配置 ====================
    # 先在固定场景预训练，策略学会基础导航/避撞后再切换为随机场景泛化。
    CURRICULUM_LEARNING = True        # 是否启用课程学习
    CURRICULUM_PRETRAIN_EPOCHS = 50   # 固定场景预训练轮数（之后自动切换为随机场景）

    # ==================== 固定评估配置 ====================
    EVAL_INTERVAL = 10          # 每 N 个 epoch 在固定场景上评估一次
    EVAL_EPISODES = 5           # 每次评估运行的 episode 数
    EVAL_SEED = 42              # 固定评估场景的随机种子（保证跨 run 可比）
    
    # 渲染设置
    RENDER_MODE = None        # 'human' 或 None（启用/禁用渲染）
    RENDER_EVERY_STEP = False   # 是否每步都渲染（慢，但详细）
    
    # 轨迹记录设置
    TRAJECTORY_MODE = False     # 是否启用轨迹记录模式（记录每架UAV的位置和速度）
    
    @classmethod
    def get_model_save_dir(cls):
        """
        【改进】根据当前配置生成模型保存目录
        
        目录结构:
        models/
        ├── policy_transformer/
        │   ├── random_100x70x40_r4f3/
        │   │   ├── policy_*.pt
        │   │   ├── training_metrics.xlsx
        │   │   └── ...
        │   └── fixed_r4f3/
        └── policy_mlp_direct/
            └── ...
        
        Returns:
            str: 完整的保存目录路径
        """
        # 基础目录：自定义路径优先（用于实验隔离）
        if cls._custom_save_dir is not None:
            base_dir = cls._custom_save_dir
        else:
            base_dir = os.path.join('./models', f'policy_{cls.POLICY_VERSION}')
        
        # 场景标识
        ver = cls.SCENARIO_VERSION  # 'v2' or 'v3'
        if cls.USE_RANDOM_SCENE:
            # 随机场景：版本 + 圆直径 + UAV数量
            diameters = cls.CIRCLE_DIAMETERS or [cls.MAX_CIRCLE_DIAMETER,
                                                  cls.MAX_CIRCLE_DIAMETER * 0.7,
                                                  cls.MAX_CIRCLE_DIAMETER * 0.4]
            diameter_str = 'x'.join(str(int(d)) for d in diameters)
            scene_identifier = f'random_{ver}_{diameter_str}_r{cls.NUM_ROUTE_UAVS}f{cls.NUM_FREE_UAVS}'
        else:
            scene_identifier = f'fixed_{ver}_r{cls.NUM_ROUTE_UAVS}f{cls.NUM_FREE_UAVS}'
        
        model_save_dir = os.path.join(base_dir, scene_identifier)
        return model_save_dir

    @classmethod
    def set_save_dir(cls, directory):
        """[新增] 设置自定义模型保存目录（用于实验隔离）。置 None 恢复默认。"""
        cls._custom_save_dir = directory

    @classmethod
    def update(cls, config_dict):
        """[新增] 批量更新配置属性。传入 dict，key 为属性名，value 为新值。"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Config 没有属性 '{key}'")

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 70)
        print("📋 多智能体 PPO 训练配置")
        print("=" * 70)
        
        print("\n🤖 策略配置:")
        print(f"  ├─ 策略版本: {cls.POLICY_VERSION}")
        if cls.POLICY_VERSION == 'transformer':
            print(f"  ├─ 嵌入维度: {cls.EMBED_DIM}")
            print(f"  ├─ 最大自由UAV数: {cls.MAX_FREE_UAVS}")
            print(f"  ├─ Transformer 层数: {cls.NUM_LAYERS}")
            print(f"  └─ MLP 隐层: {cls.MLP_HIDDEN}")
        else:
            print(f"  └─ MLP 隐层: {cls.MLP_HIDDEN}")
        
        ver_desc = {'v2': '超大圆+大中小圆（高度分离）', 'v3': '单圆（所有UAV同层）'}
        print("\n🌍 场景配置:")
        print(f"  ├─ 场景版本: {cls.SCENARIO_VERSION} — {ver_desc.get(cls.SCENARIO_VERSION, '')}")
        print(f"  ├─ 场景模式: {'随机同心圆' if cls.USE_RANDOM_SCENE else '固定场景'}")
        print(f"  ├─ 航路 UAV 数: {cls.NUM_ROUTE_UAVS}")
        print(f"  ├─ 自由 UAV 数: {cls.NUM_FREE_UAVS}")
        if cls.USE_RANDOM_SCENE:
            print(f"  ├─ 最大圆直径: {cls.MAX_CIRCLE_DIAMETER} 米")
            diameters = cls.CIRCLE_DIAMETERS or [cls.MAX_CIRCLE_DIAMETER,
                                                   cls.MAX_CIRCLE_DIAMETER * 0.7,
                                                   cls.MAX_CIRCLE_DIAMETER * 0.4]
            print(f"  └─ 圆直径: {diameters}")
        
        print("\n📊 训练超参数:")
        print(f"  ├─ 训练轮数: {cls.TRAIN_EPOCHS}")
        print(f"  ├─ 每轮步数: {cls.STEPS_PER_EPOCH}")
        print(f"  ├─ 最大步长: {cls.MAX_EP_LEN}")
        print(f"  ├─ 学习率: {cls.LEARNING_RATE}")
        print(f"  ├─ γ (折扣因子): {cls.GAMMA}")
        print(f"  ├─ λ (GAE 参数): {cls.LAM}")
        print(f"  └─ 设备: {cls.DEVICE}")
        
        print("\n💾 检查点:")
        print(f"  ├─ 保存目录: {cls.SAVE_DIR}")
        print(f"  ├─ 保存间隔: {cls.SAVE_INTERVAL} epochs")
        print(f"  ├─ 恢复 checkpoint: {cls.RESUME_FROM}")
        print(f"  └─ 渲染间隔: {cls.RENDER_INTERVAL} epochs")
        
        print("=" * 70)


class ExperimentConfigs:
    """预设的实验配置 - 方便快速切换"""
    
    @staticmethod
    def transformer_random_scene():
        """推荐配置：Transformer + 随机同心圆场景"""
        Config.POLICY_VERSION = 'transformer'
        Config.USE_RANDOM_SCENE = True
        Config.MAX_CIRCLE_DIAMETER = 100
        Config.CIRCLE_DIAMETERS = [100, 70, 40]
        Config.TRAIN_EPOCHS = 100
        Config.print_config()
    
    @staticmethod
    def transformer_fixed_scene():
        """Transformer + 固定场景（测试）"""
        Config.POLICY_VERSION = 'transformer'
        Config.USE_RANDOM_SCENE = False
        Config.TRAIN_EPOCHS = 10
        Config.print_config()
    
    @staticmethod
    def mlp_direct_random_scene():
        """MLP 直接版本 + 随机场景（对比实验）"""
        Config.POLICY_VERSION = 'mlp_direct'
        Config.USE_RANDOM_SCENE = True
        Config.print_config()
    
    @staticmethod
    def mlp_direct_fixed_scene():
        """MLP 直接版本 + 固定场景（对比实验）"""
        Config.POLICY_VERSION = 'mlp_direct'
        Config.USE_RANDOM_SCENE = False
        Config.print_config()


if __name__ == '__main__':
    print("\n🎯 可用的实验配置预设:\n")
    print("1️⃣  ExperimentConfigs.transformer_random_scene()")
    print("    → 推荐配置：Transformer + 随机同心圆场景\n")
    
    print("2️⃣  ExperimentConfigs.transformer_fixed_scene()")
    print("    → Transformer + 固定场景（快速测试）\n")
    
    print("3️⃣  ExperimentConfigs.mlp_direct_random_scene()")
    print("    → 对比实验：MLP 直接版 + 随机场景\n")
    
    print("4️⃣  ExperimentConfigs.mlp_direct_fixed_scene()")
    print("    → 对比实验：MLP 直接版 + 固定场景\n")
    
    print("\n📝 使用示例：\n")
    print("from config import Config, ExperimentConfigs")
    print("ExperimentConfigs.transformer_random_scene()")
    print("# 现在 Config 包含所有配置参数\n")
    
    print("\n⚙️  手动配置示例：\n")
    print("from config import Config")
    print("Config.POLICY_VERSION = 'transformer'")
    print("Config.USE_RANDOM_SCENE = True")
    print("Config.print_config()\n")
    
    # 演示默认配置
    print("\n当前默认配置：")
    Config.print_config()
