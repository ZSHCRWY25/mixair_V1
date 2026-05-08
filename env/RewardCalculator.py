import numpy as np

class RewardCalculator:
    @staticmethod
    def compute(route_uavs, prev_progress=None, prev_actions=None, 
                current_actions=None,
                reached_goals=None,   # 新增传入
                arrive_flags=None,     # 新增传入
                collision_uav_ids=None,
                punished_collision_ids=None,  # 【新增】本回合已经惩罚过的ID集合
                max_speed=2.0, waypoint_rewards=None, num_waypoints=None, ttc_threshold=5.0,
                collision_penalty=-100.0, waypoint_coef=1.0, terminal_coef=2.0, progress_scale=0.5,
                min_ttc_route_list=None, min_ttc_free_list=None,
                ttc_route_lateral_list=None, reset_mask=None):  # [fix] 显式传递 TTC 信息和重置掩码
        """
        【改进】细化的多目标奖励计算函数
        
        【改进1】碰撞惩罚：对具体参与碰撞的UAV惩罚，而非所有UAV
        【改进2】航点奖励：动态根据航点数量计算
        【改进3】进度奖励：增加进度激励权重
        
        Args:
            route_uavs: 航路无人机列表
            prev_progress: 上一时刻的进度
            prev_actions: 上一时刻的动作
            collision_uav_ids: 发生碰撞的UAV ID列表 (【改进】替代布尔collision参数)
            all_arrived: 是否全部到达
            max_speed: 最大预设速度 (参考速度)
            waypoint_rewards: 航点奖励列表（遗留，优先使用num_waypoints）
            num_waypoints: 航点总数（【改进】用于动态计算）
            ttc_threshold: TTC阈值
            collision_penalty: 碰撞惩罚值
            waypoint_coef: 航点奖励系数
            terminal_coef: 终点奖励系数
            progress_scale: 进度奖励缩放系数
        """
        # 【改进】初始化碰撞ID集合
        if collision_uav_ids is None:
            collision_uav_ids = set()
        else:
            collision_uav_ids = set(collision_uav_ids)
        
        # 【改进】用于本回合已经惩罚过的ID，避免重复惩罚
        if punished_collision_ids is None:
            punished_collision_ids = set()


        rewards = []
        for i, uav in enumerate(route_uavs):
            reward = 0.0

            # 【改进】碰撞惩罚：仅惩罚参与碰撞的UAV，且每个UAV每回合只惩罚一次
            if uav.id in collision_uav_ids and uav.id not in punished_collision_ids:
                rewards.append(collision_penalty)
                punished_collision_ids.add(uav.id)   # 标记为已惩罚
                continue
            elif uav.id in collision_uav_ids:
                # 已经惩罚过，本次不重复惩罚，但继续计算其他奖励（避免死循环）
                # 可以给0或正常计算，这里选择跳过惩罚后正常计算其他奖励
                pass

            # 1. 【改进】运行进度奖励：增加激励权重
            if prev_progress is not None:
                gain = uav.progress - prev_progress[i]
                # 【改进】进度奖励系数增加
                if gain > 0.1:  # 显著进度
                    reward += 0.2 * progress_scale  # 增加到 0.2 * scale
                elif gain > 0.0:  # 正常进度
                    reward += 0.1 * progress_scale * (gain / 0.1)  # 增加到 0.1 * scale
                # 如果没有进度，没有进度奖励（不惩罚，但也不奖励）

            # 2. 【改进】细化的速度奖励策略
            # 目标：在维持稳定速度和节能之间平衡
            current_speed = np.linalg.norm(uav.vel)
            
            # 定义三个速度区间
            target_speed = max_speed * 0.8  # 目标速度为最大速度的80%
            optimal_lower = max_speed * 0.6  # 最优下界
            optimal_upper = max_speed * 1.0  # 最优上界
            
            if current_speed < 0.01:
                # 停止或极慢：轻微惩罚（不要完全停止）
                speed_reward = -0.02
            elif optimal_lower <= current_speed <= optimal_upper:
                # 【关键】最优区间：获得正奖励
                # 越接近目标速度，奖励越高
                distance_to_target = abs(current_speed - target_speed)
                max_distance = target_speed - optimal_lower
                if max_distance > 0:
                    normalized_distance = distance_to_target / max_distance
                    speed_reward = 0.1 * (1.0 - normalized_distance)  # [0, 0.1]
                else:
                    speed_reward = 0.1
            elif current_speed < optimal_lower:
                # 速度偏低：小幅鼓励加速
                deficit = optimal_lower - current_speed
                speed_reward = 0.02 * (1.0 - deficit / optimal_lower)  # [0, 0.02]
            else:
                # current_speed > optimal_upper
                # 速度过高：惩罚超速
                excess = current_speed - optimal_upper
                speed_penalty = -0.05 * min(excess / optimal_upper, 1.0)  # [-0.05, 0]
                speed_reward = speed_penalty
            
            reward += speed_reward

            # 3. 动作平滑奖励
            # [fix] 若 UAV 刚被重置，跳过平滑奖励（prev_actions 已被清零，参考系无意义）
            skip_smoothness = (reset_mask is not None and reset_mask[i])
            if not skip_smoothness and prev_actions is not None:
                action_change = abs(current_actions[i] - prev_actions[i])
                if action_change < 0.1:
                    # 小幅动作变化：给予小额奖励
                    smoothness_reward = 0.01 * (1.0 - action_change / 0.1)  # [0, 0.01]
                else:
                    # 大幅动作变化：轻微惩罚
                    smoothness_penalty = -0.02 * min(action_change / 0.5, 1.0)  # [0, -0.02]
                    smoothness_reward = smoothness_penalty
                reward += smoothness_reward

            # 4. 【改进】到点奖励：动态根据航点数计算
            if reached_goals[i]:
                if hasattr(uav, 'waypoint_progress') and num_waypoints is not None:
                    # 【新】动态计算航点奖励
                    waypoint_idx = uav.waypoint_progress
                    if waypoint_idx < num_waypoints:
                        # 动态计算: reward[i] = q * m / (i+1)
                        # 其中 q=waypoint_coef, m=num_waypoints, i=waypoint_idx (0-indexed)
                        waypoint_reward = waypoint_coef * num_waypoints / (waypoint_idx + 1)
                        reward += waypoint_reward

                elif waypoint_rewards and hasattr(uav, 'waypoint_progress'):
                    # 【兼容】使用旧的固定列表
                    waypoint_idx = uav.waypoint_progress
                    if waypoint_idx < len(waypoint_rewards):
                        reward += waypoint_rewards[waypoint_idx]

            # 5. TTC 惩罚 + 侧向优先权（打破共享策略的对称困境）
            #
            # 问题根源：共享策略在 RR 冲突中两架 UAV 观测对称，梯度方向相同
            #          → 双方同时减速 → 同时到达交叉点 → 碰撞
            #
            # 解决方案：利用侧向相对位置 (rel_pos_lat) 给不同角色不同强度惩罚
            #   rel_pos_lat > 0 → 威胁 UAV 在我左侧 → 我是右侧来车 → 我有优先权
            #                     惩罚因子 < 1.0（鼓励保持速度通过）
            #   rel_pos_lat < 0 → 威胁 UAV 在我右侧 → 我是左侧来车 → 我应让行
            #                     惩罚因子 > 1.0（鼓励减速让行）
            #
            # 使用 tanh 平滑过渡，避免符号突变导致梯度抖动
            # priority_factor ∈ [0.4, 1.6]：右侧优先 0.4x，左侧让行 1.6x

            # [fix] 使用传入的 ttc_threshold 参数，替代硬编码 5.0
            TTC_WARN = ttc_threshold
            MAX_PEN_RR  = abs(collision_penalty) * 0.15
            MAX_PEN_RF  = abs(collision_penalty) * 0.08

            # [fix] 优先使用显式传入的 TTC 列表，回退到 UAV 属性（兼容旧代码）
            t_rr = (min_ttc_route_list[i] if min_ttc_route_list is not None
                    else getattr(uav, 'min_ttc_route', None))
            if t_rr is not None and t_rr < TTC_WARN:
                danger = 1.0 - t_rr / TTC_WARN
                # 侧向优先权因子：tanh 平滑，尺度 10m
                lateral = (ttc_route_lateral_list[i] if ttc_route_lateral_list is not None
                           else getattr(uav, 'ttc_route_lateral', None))
                if lateral is not None:
                    # lateral > 0 (威胁在左, 我有优先权) → factor → 0.4
                    # lateral < 0 (威胁在右, 我应让行)   → factor → 1.6
                    priority_factor = 1.0 + 0.6 * float(np.tanh(-lateral / 10.0))
                else:
                    priority_factor = 1.0
                reward -= MAX_PEN_RR * danger * priority_factor

            # [fix] 优先使用显式传入的 TTC 列表，回退到 UAV 属性
            t_rf = (min_ttc_free_list[i] if min_ttc_free_list is not None
                    else getattr(uav, 'min_ttc_free', None))
            if t_rf is not None and t_rf < TTC_WARN:
                danger = 1.0 - t_rf / TTC_WARN
                reward -= MAX_PEN_RF * danger


        # 6. 【改进】到终点奖励
        # if all_arrived:
        #     if num_waypoints is not None:
        #         # 【新】动态计算: terminal_reward = b * m
        #         terminal_reward = terminal_coef * num_waypoints
        #     else:
        #         # 【兼容】使用固定值
        #         terminal_reward = 100.0
            
        #     for i in range(len(rewards)):
        #         rewards[i] += terminal_reward
            if arrive_flags[i]:
                if num_waypoints is not None:
                    reward += terminal_coef * num_waypoints
                else:
                    reward += 50.0

            # 奖励列表
            rewards.append(reward)

        return rewards