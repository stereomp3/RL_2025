import gymnasium as gym
import numpy as np


class RacecarRewardWrapper(gym.Wrapper):
    """
    自定義獎勵機制:
    1. Motor Reward: action[0] (-1~1)
    2. Bang-Bang Penalty: -0.1 * |action - last_action|
    3. Progress Reward: 1000 * delta_progress
    4. Wall Collision: -100
    """

    def __init__(self, env, stability_weight=0.1, collision_penalty=-100.0, survival_reward=0.0):
        super().__init__(env)
        self.stability_weight = stability_weight  # 對應 bang-bang 的係數 0.1
        self.collision_penalty = collision_penalty  # 對應 -100

        # 記錄變數
        self.last_action = None
        self.last_total_progress = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 初始化
        self.last_action = np.zeros(self.action_space.shape)
        self.last_total_progress = info.get('lap', 0) + info.get('progress', 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- 1. Motor Reward (動力獎勵) ---
        # 根據 env.py: actions = [motor, steering]
        # 所以 action[0] 是 motor
        motor_reward = float(action[0])

        # --- 2. Bang-Bang Penalty (動作平滑度) ---
        # -0.1 * sum(abs(current - last))
        action_diff = np.abs(action - self.last_action)
        bang_bang_penalty = -self.stability_weight * np.sum(action_diff)

        # --- 3. Progress Reward (進度獎勵) ---
        # 計算總進度 (圈數 + 本圈百分比)
        current_total_progress = info.get('lap', 0) + info.get('progress', 0)

        # 計算增量
        progress_delta = current_total_progress - self.last_total_progress

        # 異常保護: 如果換圈時產生負值，則視為 0
        if progress_delta < -0.5:
            progress_delta = 0.0

        progress_reward = 1000.0 * progress_delta

        # --- 4. Wall Collision Penalty (撞牆懲罰) ---
        collision_reward = 0.0
        # RaceEnv 的 info 通常包含 'wall_collision'
        if info.get('wall_collision', False):
            collision_reward = self.collision_penalty  # -100
            terminated = True  # 撞牆直接結束

        # --- 計算總分 ---
        # 覆蓋原始 reward
        new_reward = motor_reward + bang_bang_penalty + progress_reward + collision_reward

        # 更新狀態
        self.last_action = np.copy(action)
        self.last_total_progress = current_total_progress

        return obs, new_reward, terminated, truncated, info