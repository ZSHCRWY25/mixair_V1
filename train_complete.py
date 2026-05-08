#!/usr/bin/env python
"""
多智能体 PPO 完整训练脚本
支持：
- 两种策略版本切换（Transformer 创新版 vs MLP 直接版）
- 随机同心圆场景（每轮重新生成）或固定场景
- 完整的指标记录和模型保存
"""

# 修复 OpenMP 运行时冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
import time
import matplotlib

# 【渲染后端选择】
# - 如果需要实时渲染窗口，改成: matplotlib.use('TkAgg') 或 'Qt5Agg'
# - 如果只做离线训练，保留: matplotlib.use('Agg') 可提高性能
# 为了支持渲染，默认优先尝试TkAgg（需要系统支持Tkinter）
try:
    matplotlib.use('TkAgg')  # 交互式后端，支持窗口显示
except:
    try:
        matplotlib.use('Qt5Agg')  # 备选：Qt5后端
    except:
        matplotlib.use('Agg')  # 最后降级到Agg非交互后端

import matplotlib.pyplot as plt
import pandas as pd
from env.uavenv import UAVEnv
from policy.ppo import PPO, PPOBuffer
from config import Config, ExperimentConfigs


def create_policy(policy_version='transformer'):
    """根据配置创建策略网络"""
    if policy_version == 'transformer':
        from policy.transformer_policy_v2 import TransformerPolicyV2
        print("✨ 使用 Transformer 创新版本策略")
        return TransformerPolicyV2(
            self_state_dim=Config.SELF_STATE_DIM,
            other_route_dim=Config.OTHER_ROUTE_DIM,
            free_uav_dim=Config.FREE_UAV_DIM,
            max_free_uavs=Config.MAX_FREE_UAVS,
            embed_dim=Config.EMBED_DIM,
            nhead=Config.NHEAD,
            num_layers=Config.NUM_LAYERS,
            mlp_hidden=Config.MLP_HIDDEN,
            device=Config.DEVICE
        )
    elif policy_version == 'mlp_direct':
        from policy.transformer_policy_v2 import MLPPolicyDirect
        print("🔷 使用 MLP 直接版本策略（对比实验）")
        obs_dim = Config.SELF_STATE_DIM + Config.OTHER_ROUTE_DIM + Config.MAX_FREE_UAVS * Config.FREE_UAV_DIM
        return MLPPolicyDirect(
            obs_dim=obs_dim,
            act_dim=1,
            hidden_sizes=Config.MLP_HIDDEN,
            device=Config.DEVICE
        )
    else:
        raise ValueError(f"未知的策略版本: {policy_version}")


class TrainerWithDynamicScene:
    """支持动态场景生成的训练器"""
    
    def __init__(self, env, policy, device='cpu', trajectory_mode=False):
        self.env = env
        self.policy = policy
        self.device = torch.device(device) if isinstance(device, str) else device
        self.trajectory_mode = trajectory_mode  # 轨迹记录模式
        self.start_time = None  # 训练开始时间，用于时间估计
        self.ppo = PPO(policy,
                       pi_lr=Config.LEARNING_RATE,
                       vf_lr=Config.LEARNING_RATE,
                       gamma=Config.GAMMA,
                       lam=Config.LAM,
                       clip_ratio=Config.CLIP_RATIO,
                       target_kl=Config.TARGET_KL,
                       train_pi_iters=Config.TRAIN_PI_ITERS,
                       train_v_iters=Config.TRAIN_V_ITERS,
                       max_grad_norm=Config.MAX_GRAD_NORM,
                       use_gpu=(self.device.type == 'cuda'),
                       entropy_coef=Config.ENTROPY_COEF)
        
        # 指标记录
        self.epoch_returns = []
        self.epoch_collisions = []
        self.epoch_lengths = []
        self.epoch_times = []
        self.epoch_avg_total_returns = []
        self.epoch_avg_avg_returns = []
        self.epoch_avg_max_returns = []
        self.epoch_avg_min_returns = []
        self.epoch_avg_std_returns = []
        self.epoch_avg_returns = []
        self.epoch_collision_counts = []
        self.epoch_route_route_collisions = []
        self.epoch_route_free_collisions = []
        self.epoch_avg_lengths = []
        self.epoch_avg_speeds = []
        self.epoch_conflict_counts = []
        self.epoch_rr_conflict_counts = []
        self.epoch_rf_conflict_counts = []
        self.epoch_collision_avoidance_rates = []
        self.epoch_rr_avoidance_rates = []
        self.epoch_rf_avoidance_rates = []

        # 固定场景评估指标
        self.eval_epochs = []
        self.eval_avg_returns = []
        self.eval_avg_avoidances = []
        self.eval_std_returns = []

        # 生成固定评估场景（seeded，保证跨run可比）
        rng_state = np.random.get_state()
        np.random.seed(Config.EVAL_SEED)
        from env.ScenarioGeneratorV2 import ScenarioGeneratorV2
        from env.ScenarioGeneratorV3 import ScenarioGeneratorV3
        _GenCls = ScenarioGeneratorV3 if Config.SCENARIO_VERSION == 'v3' else ScenarioGeneratorV2
        _eval_gen = _GenCls(env.num_route_uavs, env.num_free_uavs,
                            max_circle_diameter=Config.MAX_CIRCLE_DIAMETER,
                            circle_diameters=Config.CIRCLE_DIAMETERS)
        self.eval_route_configs, self.eval_free_configs = _eval_gen.generate()
        np.random.set_state(rng_state)

        # 轨迹记录（如果启用）
        if self.trajectory_mode:
            self.trajectory_data = []  # 存储每架无人机的轨迹
    
    def train(self, train_epochs=40, steps_per_epoch=1000, max_ep_len=500, 
              gamma=0.99, lam=0.95, regenerate_scene_each_epoch=True, render_every_step=False):
        """
        训练循环
        
        Args:
            regenerate_scene_each_epoch: 每个 epoch 是否重新生成场景
            render_every_step: 是否每步都渲染（慢，但详细）
        """
        num_agents = self.env.num_route_uavs
        obs_dim = self.env.observation_space.shape[-1]
        
        print(f"\n🎯 开始训练")
        print(f"   策略版本: {Config.POLICY_VERSION}")
        print(f"   场景模式: {'随机同心圆' if Config.USE_RANDOM_SCENE else '固定'}")
        print(f"   智能体数: {num_agents}")
        print(f"   观测维度: {obs_dim}")
        print(f"   训练轮数: {train_epochs}")
        print(f"   设备: {self.device}")
        print("=" * 70)
        
        self.start_time = time.time()  # 记录训练开始时间
        
        for epoch in range(train_epochs):
            epoch_start_time = time.time()
            
            # 检查是否需要渲染这个epoch
            render_this_epoch = (epoch + 1) % Config.RENDER_INTERVAL == 0 and Config.RENDER_MODE == 'human'
            
            # 如果需要渲染，临时启用每步渲染
            if render_this_epoch:
                original_render_mode = self.env.render_mode
                self.env.render_mode = 'human'
                print(f"🎨 Epoch {epoch+1}: 启用渲染模式")
            else:
                original_render_mode = self.env.render_mode
                if not render_every_step:
                    self.env.render_mode = None  # 禁用渲染以提高速度
            
            # 课程学习：前 N 个 epoch 用固定场景预训练，之后切换随机场景
            in_pretrain = (Config.CURRICULUM_LEARNING and
                           Config.USE_RANDOM_SCENE and
                           epoch < Config.CURRICULUM_PRETRAIN_EPOCHS)

            if in_pretrain:
                # 注入固定评估场景（与 eval 相同的 seed 场景）
                self.env._last_route_configs = self.eval_route_configs
                self.env._last_free_configs  = self.eval_free_configs
                obs_list, _ = self.env.reset(regenerate_scene=False)
                if epoch == 0:
                    print(f"  [课程学习] 预训练阶段：使用固定场景，共 {Config.CURRICULUM_PRETRAIN_EPOCHS} 轮")
            elif epoch == Config.CURRICULUM_PRETRAIN_EPOCHS and Config.CURRICULUM_LEARNING:
                print(f"  [课程学习] 切换到随机场景训练")
                obs_list, _ = self.env.reset(regenerate_scene=True)
            elif regenerate_scene_each_epoch and Config.USE_RANDOM_SCENE:
                obs_list, _ = self.env.reset(regenerate_scene=True)
            else:
                obs_list, _ = self.env.reset(regenerate_scene=False)
            
            # 创建缓冲区
            buf = PPOBuffer(obs_dim, 1, steps_per_epoch, gamma, lam, num_agents)
            
            ep_ret = np.zeros(num_agents)
            ep_len = 0
            ep_collisions = 0
            ep_route_route_collisions = 0
            ep_route_free_collisions = 0
            ep_conflict_count = 0
            ep_rr_conflict_count = 0
            ep_rf_conflict_count = 0
            epoch_episode_total_returns = []  # 【新增】总回报列表
            epoch_episode_avg_returns = []   # 【新增】平均回报列表
            epoch_episode_max_returns = []   # 【新增】最大回报列表
            epoch_episode_min_returns = []   # 【新增】最小回报列表
            epoch_episode_std_returns = []   # 【新增】标准差回报列表
            epoch_episode_returns = []       # 【新增】episode 回报列表
            epoch_episode_collisions = []
            epoch_episode_route_route_collisions = []
            epoch_episode_route_free_collisions = []
            epoch_episode_conflict_counts = []
            epoch_episode_rr_conflict_counts = []
            epoch_episode_rf_conflict_counts = []
            epoch_episode_lengths = []
            epoch_speeds = []  # 新增：记录每步的速度
            
            # 轨迹记录初始化
            if self.trajectory_mode:
                episode_trajectories = [[] for _ in range(num_agents)]
            
            # 采集经验（eval模式：关闭dropout，确保logp_old与更新时第一轮logp_new一致）
            self.policy.eval()
            for step in range(steps_per_epoch):
                actions = []
                logps = []
                values = []
                
                # 为每个智能体采样动作
                for i in range(num_agents):
                    obs_t = torch.as_tensor(obs_list[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        dist = self.policy.get_dist(obs_t)
                        action = dist.sample()
                        logp = dist.log_prob(action).sum(dim=-1)
                        _, value = self.policy.forward(obs_t)
                        actions.append(action.item())
                        logps.append(logp.item())
                        values.append(value.squeeze().item())
                
                # 执行环境步
                next_obs_list, reward_list, terminated, truncated, info = self.env.step(np.array(actions))
                done = terminated or truncated
                
                # 记录轨迹和航路无人机平均速度
                route_uavs = self.env.manager.get_route_uavs()
                route_speeds = [np.linalg.norm(uav.vel) for uav in route_uavs]
                avg_route_speed = float(np.mean(route_speeds)) if route_speeds else 0.0
                epoch_speeds.append(avg_route_speed)
                if self.trajectory_mode:
                    for i in range(num_agents):
                        pos = route_uavs[i].pos.copy()
                        vel = route_uavs[i].vel.copy()
                        speed = np.linalg.norm(vel)
                        episode_trajectories[i].append({
                            'step': step,
                            'position': pos,
                            'velocity': vel,
                            'speed': speed
                        })
                
                # 存储经验
                buf.store(obs_list, actions, reward_list, values, logps)
                obs_list = next_obs_list
                ep_ret += np.array(reward_list)
                ep_len += 1

                # 处理单架无人机轨迹终止（碰撞或到达目的地）
                # finish_path_agent 结束该 agent 的 GAE 计算；ep_ret 保留累积值，
                # 不清零——保证 episode_return 反映完整 episode 的真实表现
                per_agent_done = info.get('per_agent_done', [False] * num_agents)
                for i, agent_done in enumerate(per_agent_done):
                    if agent_done and not done:
                        buf.finish_path_agent(i, 0.0)
                        # 注意：ep_ret[i] 不再清零，保留累积回报供 episode_return 使用

                # 统计碰撞
                if 'route_route_collisions' in info and info['route_route_collisions'] > 0:
                    ep_route_route_collisions += info['route_route_collisions']
                if 'route_free_collisions' in info and info['route_free_collisions'] > 0:
                    ep_route_free_collisions += info['route_free_collisions']
                ep_collisions += info.get('route_route_collisions', 0) + info.get('route_free_collisions', 0)
                
                ep_conflict_count    += info.get('conflict_count',    0)
                ep_rr_conflict_count += info.get('rr_conflict_count', 0)
                ep_rf_conflict_count += info.get('rf_conflict_count', 0)
                
                # 处理回合结束
                if done:
                    buf.finish_path([0.0] * num_agents)
                    
                    # 【改进】计算多种回报统计指标
                    ep_ret_array = np.array(ep_ret)  # 转换为数组便于统计
                    
                    # 总回报：所有步所有UAV的奖励总和
                    episode_total_return = np.sum(ep_ret_array)
                    
                    # 平均回报：每步每UAV的平均奖励
                    episode_avg_return = np.mean(ep_ret_array)
                    
                    # 最大回报：单步最大奖励
                    episode_max_return = np.max(ep_ret_array)
                    
                    # 最小回报：单步最小奖励  
                    episode_min_return = np.min(ep_ret_array)
                    
                    # 标准化回报：奖励的标准差
                    episode_std_return = np.std(ep_ret_array)
                    
                    episode_collisions = ep_collisions
                    episode_route_route_collisions = ep_route_route_collisions
                    episode_route_free_collisions = ep_route_free_collisions
                    episode_length = ep_len
                    
                    # 【新增】计算避撞成功率
                    # 成功率 = 1 - (实际碰撞 / 潜在冲突)
                    # 如果没有潜在冲突，则成功率为 100%
                    if ep_conflict_count > 0:
                        collision_avoidance_rate = max(0.0, 1.0 - (ep_collisions / ep_conflict_count))
                    else:
                        collision_avoidance_rate = 1.0  # 无冲突时成功率为100%

                    episode_return = np.sum(ep_ret)  # 【修复】定义 episode_return 为 episode 总回报
                    
                    self.epoch_returns.append(episode_return)
                    self.epoch_collisions.append(episode_collisions)
                    self.epoch_lengths.append(episode_length)

                    epoch_episode_returns.append(episode_return)
                    epoch_episode_collisions.append(episode_collisions)
                    epoch_episode_route_route_collisions.append(episode_route_route_collisions)
                    epoch_episode_route_free_collisions.append(episode_route_free_collisions)
                    epoch_episode_conflict_counts.append(ep_conflict_count)
                    epoch_episode_rr_conflict_counts.append(ep_rr_conflict_count)
                    epoch_episode_rf_conflict_counts.append(ep_rf_conflict_count)

                    epoch_episode_lengths.append(episode_length)
                    
                    # 【新增】保存多种回报统计指标
                    epoch_episode_total_returns.append(episode_total_return)
                    epoch_episode_avg_returns.append(episode_avg_return)
                    epoch_episode_max_returns.append(episode_max_return)
                    epoch_episode_min_returns.append(episode_min_return)
                    epoch_episode_std_returns.append(episode_std_return)

                    # 保存轨迹数据
                    if self.trajectory_mode:
                        self.trajectory_data.append({
                            'epoch': epoch,
                            'episode': len(epoch_episode_returns),
                            'trajectories': episode_trajectories
                        })

                    obs_list, _ = self.env.reset(regenerate_scene=False)  # episode内复用同一场景
                    ep_ret = np.zeros(num_agents)
                    ep_len = 0
                    ep_collisions = 0
                    ep_route_route_collisions = 0
                    ep_route_free_collisions = 0
                    ep_conflict_count    = 0
                    ep_rr_conflict_count = 0
                    ep_rf_conflict_count = 0

                    # 重置轨迹记录
                    if self.trajectory_mode:
                        episode_trajectories = [[] for _ in range(num_agents)]
            
            # 计算最后价值做 bootstrap
            last_vals = []
            for i in range(num_agents):
                obs_t = torch.as_tensor(obs_list[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    _, value = self.policy.forward(obs_t)
                    last_val = 0 if done else value.squeeze().item()
                last_vals.append(last_val)
            buf.finish_path(last_vals)
            
            # PPO 更新（切换回train模式）
            self.policy.train()
            data = buf.get()
            update_info = self.ppo.update(data)
            
            # 输出统计信息
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            if len(epoch_episode_returns) > 0:
                # 【改进】计算多种回报统计指标
                epoch_total_returns = np.array(epoch_episode_total_returns)
                epoch_avg_returns = np.array(epoch_episode_avg_returns)
                epoch_max_returns = np.array(epoch_episode_max_returns)
                epoch_min_returns = np.array(epoch_episode_min_returns)
                epoch_std_returns = np.array(epoch_episode_std_returns)
                
                # 每轮的平均统计
                epoch_avg_total_return = np.mean(epoch_total_returns)
                epoch_avg_avg_return = np.mean(epoch_avg_returns)
                epoch_avg_max_return = np.mean(epoch_max_returns)
                epoch_avg_min_return = np.mean(epoch_min_returns)
                epoch_avg_std_return = np.mean(epoch_std_returns)
                epoch_avg_return = np.mean(epoch_episode_returns)

                epoch_total_collisions = np.sum(epoch_episode_collisions)
                epoch_total_route_route_collisions = np.sum(epoch_episode_route_route_collisions)
                epoch_total_route_free_collisions = np.sum(epoch_episode_route_free_collisions)
                epoch_avg_length = np.mean(epoch_episode_lengths)
                epoch_avg_speed = np.mean(epoch_speeds) if epoch_speeds else 0.0
                
                epoch_total_conflicts    = np.sum(epoch_episode_conflict_counts)
                epoch_total_rr_conflicts = np.sum(epoch_episode_rr_conflict_counts)
                epoch_total_rf_conflicts = np.sum(epoch_episode_rf_conflict_counts)

                def _avoidance(collisions, conflicts):
                    return max(0.0, 1.0 - collisions / conflicts) if conflicts > 0 else 1.0

                epoch_avg_collision_avoidance_rate = _avoidance(epoch_total_collisions, epoch_total_conflicts)
                epoch_rr_avoidance_rate = _avoidance(epoch_total_route_route_collisions, epoch_total_rr_conflicts)
                epoch_rf_avoidance_rate = _avoidance(epoch_total_route_free_collisions,  epoch_total_rf_conflicts)
            else:
                epoch_avg_total_return = 0.0
                epoch_avg_avg_return = 0.0
                epoch_avg_max_return = 0.0
                epoch_avg_min_return = 0.0
                epoch_avg_std_return = 0.0
                epoch_avg_return = 0.0
                epoch_total_collisions = 0
                epoch_total_route_route_collisions = 0
                epoch_total_route_free_collisions = 0
                epoch_avg_length = ep_len
                epoch_avg_speed = 0.0
                epoch_total_conflicts    = 0
                epoch_total_rr_conflicts = 0
                epoch_total_rf_conflicts = 0
                epoch_avg_collision_avoidance_rate = 0.0
                epoch_rr_avoidance_rate  = 0.0
                epoch_rf_avoidance_rate  = 0.0

            self.epoch_avg_total_returns.append(epoch_avg_total_return)
            self.epoch_avg_avg_returns.append(epoch_avg_avg_return)
            self.epoch_avg_max_returns.append(epoch_avg_max_return)
            self.epoch_avg_min_returns.append(epoch_avg_min_return)
            self.epoch_avg_std_returns.append(epoch_avg_std_return)
            self.epoch_collision_counts.append(epoch_total_collisions)
            self.epoch_route_route_collisions.append(epoch_total_route_route_collisions)
            self.epoch_route_free_collisions.append(epoch_total_route_free_collisions)
            self.epoch_avg_lengths.append(epoch_avg_length)
            self.epoch_avg_speeds.append(epoch_avg_speed)
            self.epoch_avg_returns.append(epoch_avg_return)  # 【新增】保存每轮平均episode回报
            
            self.epoch_conflict_counts.append(epoch_total_conflicts)
            self.epoch_rr_conflict_counts.append(epoch_total_rr_conflicts)
            self.epoch_rf_conflict_counts.append(epoch_total_rf_conflicts)
            self.epoch_collision_avoidance_rates.append(epoch_avg_collision_avoidance_rate)
            self.epoch_rr_avoidance_rates.append(epoch_rr_avoidance_rate)
            self.epoch_rf_avoidance_rates.append(epoch_rf_avoidance_rate)

            print(f"Epoch {epoch+1:3d}/{train_epochs} | {epoch_time:5.1f}s | "
                  f"Ret: {epoch_avg_return:.1f}"
                  f"[{epoch_avg_min_return:.0f}~{epoch_avg_max_return:.0f}] | "
                  f"Coll: {epoch_total_collisions:3d}"
                  f"(RR:{epoch_total_route_route_collisions} RF:{epoch_total_route_free_collisions}) | "
                  f"AvRR:{epoch_rr_avoidance_rate:.1%} AvRF:{epoch_rf_avoidance_rate:.1%} | "
                  f"π:{update_info['pi_loss']:6.4f} V:{update_info['v_loss']:6.4f} "
                  f"H:{update_info['entropy']:.3f} KL:{update_info['kl']:.4f}")

            # 固定场景评估（每 EVAL_INTERVAL 个 epoch 一次）
            if (epoch + 1) % Config.EVAL_INTERVAL == 0:
                eval_result = self._run_eval(num_agents, Config.EVAL_EPISODES)
                self.eval_epochs.append(epoch + 1)
                self.eval_avg_returns.append(eval_result['avg_return'])
                self.eval_avg_avoidances.append(eval_result['avg_avoidance'])
                self.eval_std_returns.append(eval_result['std_return'])
                print(f"  ▶ [固定场景评估] Ret={eval_result['avg_return']:6.2f}±{eval_result['std_return']:.2f} | "
                      f"Avoid={eval_result['avg_avoidance']:.2%} | "
                      f"RR={eval_result['rr_collisions']:.1f} RF={eval_result['rf_collisions']:.1f}")

            # 估计剩余训练时间
            elapsed_time = time.time() - self.start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = train_epochs - (epoch + 1)
            estimated_remaining_time = remaining_epochs * avg_time_per_epoch
            estimated_total_hours = estimated_remaining_time / 3600
            print(f"预计剩余时间: {estimated_total_hours:.1f} 小时")
            
            # 保存模型
            if (epoch + 1) % Config.SAVE_INTERVAL == 0:
                # 【改进】使用配置化的目录结构
                model_save_dir = Config.get_model_save_dir()
                os.makedirs(model_save_dir, exist_ok=True)
                
                save_path = os.path.join(model_save_dir, 
                                        f"policy_{Config.POLICY_VERSION}_epoch{epoch+1}.pt")
                torch.save({
                    'policy_state_dict': self.policy.state_dict(),
                    'config': {
                        'policy_version': Config.POLICY_VERSION,
                        'use_random_scene': Config.USE_RANDOM_SCENE,
                        'num_route_uavs': num_agents,
                        'num_free_uavs': Config.NUM_FREE_UAVS,
                        'obs_dim': obs_dim,
                        'circle_diameters': Config.CIRCLE_DIAMETERS,
                        'max_circle_diameter': Config.MAX_CIRCLE_DIAMETER,
                        'epoch': epoch + 1
                    }
                }, save_path)
                print(f"✅ 模型已保存: {save_path}")
            # 恢复原始渲染模式
            self.env.render_mode = original_render_mode
        print(f"✅ 训练完成！")
        print(f"   总耗时: {np.sum(self.epoch_times):.2f}s")
        print(f"   最近10次回报平均: {np.mean(self.epoch_returns[-10:]):.2f}")
        print(f"   总碰撞数: {np.sum(self.epoch_collision_counts)} | "
              f"Route-Route: {np.sum(self.epoch_route_route_collisions)} | "
              f"Route-Free: {np.sum(self.epoch_route_free_collisions)}")
        print(f"   总冲突数 (TTC<3s): {np.sum(self.epoch_conflict_counts)}")  # 【新增】
        print(f"   平均避撞成功率: {np.mean(self.epoch_collision_avoidance_rates):.2%}")  # 【新增】
        print(f"   每轮平均回报曲线已保存: {self._save_training_metrics()} ")
        print("=" * 70)

    def _run_eval(self, num_agents, n_episodes):
        """在固定评估场景上运行若干 episode，返回统计指标（不影响训练场景状态）"""
        # 保存当前训练场景配置
        saved_route = getattr(self.env, '_last_route_configs', None)
        saved_free  = getattr(self.env, '_last_free_configs',  None)

        # 注入固定评估场景
        self.env._last_route_configs = self.eval_route_configs
        self.env._last_free_configs  = self.eval_free_configs

        # 临时关闭渲染以加速评估
        saved_render = self.env.render_mode
        self.env.render_mode = None

        episode_returns    = []
        episode_rr         = []
        episode_rf         = []
        episode_avoidances = []

        for _ in range(n_episodes):
            obs_list, _ = self.env.reset(regenerate_scene=False)
            ep_ret      = np.zeros(num_agents)
            ep_rr       = 0
            ep_rf       = 0
            ep_collisions = 0
            ep_conflicts  = 0
            done = False

            while not done:
                actions = []
                with torch.no_grad():
                    for i in range(num_agents):
                        obs_t = torch.as_tensor(
                            obs_list[i], dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        dist   = self.policy.get_dist(obs_t)
                        action = dist.sample()
                        actions.append(action.item())

                next_obs_list, reward_list, terminated, truncated, info = \
                    self.env.step(np.array(actions))
                done      = terminated or truncated
                obs_list  = next_obs_list
                ep_ret   += np.array(reward_list)
                step_rr   = info.get('route_route_collisions', 0)
                step_rf   = info.get('route_free_collisions',  0)
                ep_rr        += step_rr
                ep_rf        += step_rf
                ep_collisions += step_rr + step_rf
                ep_conflicts  += info.get('conflict_count', 0)

            episode_returns.append(float(np.sum(ep_ret)))
            episode_rr.append(ep_rr)
            episode_rf.append(ep_rf)
            avoidance = (
                max(0.0, 1.0 - ep_collisions / ep_conflicts)
                if ep_conflicts > 0 else 1.0
            )
            episode_avoidances.append(avoidance)

        # 恢复训练场景和渲染模式
        if saved_route is not None:
            self.env._last_route_configs = saved_route
        if saved_free is not None:
            self.env._last_free_configs = saved_free
        self.env.render_mode = saved_render

        return {
            'avg_return':    float(np.mean(episode_returns)),
            'std_return':    float(np.std(episode_returns)),
            'avg_avoidance': float(np.mean(episode_avoidances)),
            'rr_collisions': float(np.mean(episode_rr)),
            'rf_collisions': float(np.mean(episode_rf)),
        }

    def _save_training_metrics(self):
        """【改进】保存训练过程中每轮平均回报、碰撞次数和平均速度到Excel和图片，使用配置化目录"""
        # 【改进】使用配置化的目录结构
        save_dir = Config.get_model_save_dir()
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存Excel文件
        excel_path = os.path.join(save_dir, 'training_metrics.xlsx')
        epochs = list(range(1, len(self.epoch_avg_returns) + 1))
        df = pd.DataFrame({
            'Epoch': epochs,
            'Average_Return': self.epoch_avg_returns,
            'Total_Return': self.epoch_avg_total_returns,  # 【新增】总回报
            'Avg_Return': self.epoch_avg_avg_returns,     # 【新增】平均回报
            'Max_Return': self.epoch_avg_max_returns,     # 【新增】最大回报
            'Min_Return': self.epoch_avg_min_returns,     # 【新增】最小回报
            'Std_Return': self.epoch_avg_std_returns,     # 【新增】回报标准差
            'Collision_Count': self.epoch_collision_counts,
            'Route_Route_Collisions': self.epoch_route_route_collisions,
            'Route_Free_Collisions': self.epoch_route_free_collisions,
            'Conflict_Count':    self.epoch_conflict_counts,
            'RR_Conflict_Count': self.epoch_rr_conflict_counts,
            'RF_Conflict_Count': self.epoch_rf_conflict_counts,
            'Avoidance_Rate':    self.epoch_collision_avoidance_rates,
            'RR_Avoidance_Rate': self.epoch_rr_avoidance_rates,
            'RF_Avoidance_Rate': self.epoch_rf_avoidance_rates,
            'Average_Speed': self.epoch_avg_speeds,
            'Average_Length': self.epoch_avg_lengths,
            'Training_Time': self.epoch_times
        })
        df.to_excel(excel_path, index=False)

        # 固定场景评估数据（写到独立 sheet）
        if self.eval_epochs:
            df_eval = pd.DataFrame({
                'Epoch':           self.eval_epochs,
                'Eval_Avg_Return': self.eval_avg_returns,
                'Eval_Std_Return': self.eval_std_returns,
                'Eval_Avoidance':  self.eval_avg_avoidances,
            })
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
                df_eval.to_excel(writer, sheet_name='Eval_Metrics', index=False)

        # 保存轨迹数据（如果启用）
        if self.trajectory_mode and self.trajectory_data:
            trajectory_path = os.path.join(save_dir, 'trajectories.xlsx')
            trajectory_rows = []
            for episode_data in self.trajectory_data:
                epoch = episode_data['epoch']
                episode = episode_data['episode']
                for uav_id, trajectory in enumerate(episode_data['trajectories']):
                    for point in trajectory:
                        trajectory_rows.append({
                            'Epoch': epoch,
                            'Episode': episode,
                            'UAV_ID': uav_id,
                            'Step': point['step'],
                            'Position_X': point['position'][0],
                            'Position_Y': point['position'][1],
                            'Position_Z': point['position'][2],
                            'Velocity_X': point['velocity'][0],
                            'Velocity_Y': point['velocity'][1],
                            'Velocity_Z': point['velocity'][2],
                            'Speed': point['speed']
                        })
            if trajectory_rows:
                traj_df = pd.DataFrame(trajectory_rows)
                traj_df.to_excel(trajectory_path, index=False)
        
        # 保存图片
        save_path = os.path.join(save_dir, 'training_metrics.png')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 子图1: 训练平均回报 + 固定场景评估回报（叠加）
        ax1.plot(epochs, self.epoch_avg_returns, color='tab:blue', label='Train')
        if self.eval_epochs:
            ax1.errorbar(self.eval_epochs, self.eval_avg_returns,
                         yerr=self.eval_std_returns,
                         color='tab:orange', marker='o', linestyle='--',
                         capsize=3, label='Eval (fixed scene)')
            ax1.legend(fontsize=8)
        ax1.set_title('Average Return per Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Average Return')
        ax1.grid(True, linestyle='--', alpha=0.3)

        # 子图2: RR / RF 避撞成功率（训练） + 固定场景评估避撞率
        ax2.plot(epochs, self.epoch_rr_avoidance_rates, color='tab:blue',
                 label='RR avoidance', linewidth=1.2)
        ax2.plot(epochs, self.epoch_rf_avoidance_rates, color='tab:green',
                 label='RF avoidance', linewidth=1.2)
        if self.eval_epochs:
            ax2.plot(self.eval_epochs, self.eval_avg_avoidances,
                     color='tab:orange', marker='o', linestyle='--',
                     label='Eval avoidance', markersize=4)
        ax2.set_ylim(0, 1.05)
        ax2.set_title('Collision Avoidance Rate (RR / RF / Eval)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Avoidance Rate')
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle='--', alpha=0.3)

        # 子图3: 平均速度
        ax3.plot(epochs, self.epoch_avg_speeds, color='tab:green')
        ax3.set_title('Average Speed per Epoch')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Average Speed (m/s)')
        ax3.grid(True, linestyle='--', alpha=0.3)

        # 子图4: 训练时间
        ax4.plot(epochs, self.epoch_times, color='tab:orange')
        ax4.set_title('Training Time per Epoch')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (s)')
        ax4.grid(True, linestyle='--', alpha=0.3)

        fig.suptitle('Training Metrics per Epoch')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        
        return save_path


def main():
    # 【配置选择】使用预设配置或手动配置
    
    # 方式 1: 使用预设配置（推荐）
    # ExperimentConfigs.transformer_random_scene()  # Transformer + 随机场景
    
    # 方式 2: 手动配置
    Config.POLICY_VERSION = 'transformer'
    Config.USE_RANDOM_SCENE = True
    Config.print_config()
    
    # 方式 3: 其他预设配置
    # ExperimentConfigs.mlp_direct_random_scene()   # MLP 对比
    # ExperimentConfigs.mlp_direct_fixed_scene()    # MLP + 固定场景
    
    # 创建环境
    print("\n🌍 创建环境...")
    env = UAVEnv(
        num_route_uavs=Config.NUM_ROUTE_UAVS,
        num_free_uavs=Config.NUM_FREE_UAVS,
        max_steps=Config.MAX_EP_LEN,
        render_mode=Config.RENDER_MODE if Config.RENDER_MODE else None,
        use_random_scene=Config.USE_RANDOM_SCENE,
        max_circle_diameter=Config.MAX_CIRCLE_DIAMETER,
        circle_diameters=Config.CIRCLE_DIAMETERS
    )
    
    # 创建策略
    print("\n🤖 创建策略网络...")
    policy = create_policy(Config.POLICY_VERSION)

    # 如果配置了 checkpoint，则加载已有模型权重继续训练
    if Config.RESUME_FROM is not None:
        checkpoint_path = os.path.expanduser(Config.RESUME_FROM)
        if os.path.isfile(checkpoint_path):
            print(f"🔁 加载 checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
            if 'policy_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['policy_state_dict'])
                print("✅ 成功加载策略权重。")
            else:
                raise KeyError(f"Checkpoint 中未找到 policy_state_dict: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Resume checkpoint 未找到: {checkpoint_path}")
    
    # 创建训练器
    print("\n📚 创建训练器...")
    trajectory_mode = Config.TRAJECTORY_MODE  # 从配置中读取轨迹模式
    trainer = TrainerWithDynamicScene(env, policy, device=Config.DEVICE, trajectory_mode=trajectory_mode)
    
    # 训练
    trainer.train(
        train_epochs=Config.TRAIN_EPOCHS,
        steps_per_epoch=Config.STEPS_PER_EPOCH,
        max_ep_len=Config.MAX_EP_LEN,
        gamma=Config.GAMMA,
        lam=Config.LAM,
        regenerate_scene_each_epoch=Config.USE_RANDOM_SCENE,  # 【关键】场景重生成
        render_every_step=Config.RENDER_EVERY_STEP
    )


if __name__ == '__main__':
    main()
