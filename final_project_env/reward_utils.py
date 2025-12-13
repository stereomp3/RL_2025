import gymnasium as gym
import numpy as np


class RacecarRewardWrapper(gym.Wrapper):
    """
    自定義獎勵塑形 Wrapper (Reward Shaping Wrapper)
    用於在原始環境的獎勵基礎上，增加：
    1. 穩定性懲罰 (動作平滑度)
    2. 強化的碰撞懲罰
    3. 生存獎勵 (可選)
    """

    def __init__(self, env, stability_weight=0.5, collision_penalty=-10.0, survival_reward=0.01):
        super().__init__(env)
        self.stability_weight = stability_weight
        self.collision_penalty = collision_penalty
        self.survival_reward = survival_reward

        # 記錄上一次的動作，用於計算穩定性
        self.last_action = None

    def reset(self, **kwargs):
        # 回合重置時，清空上一次的動作記錄
        self.last_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- 1. 基礎獎勵 (Progress Reward) ---
        # 原始環境通常回傳的是「進度增量 (delta progress)」，這就是我們的 Base Reward
        # 如果需要強調速度，可以將原始獎勵放大，例如: reward *= 1.0

        # --- 2. 碰撞懲罰 (Collision Penalty) ---
        # 檢查是否發生碰撞 (通常在 terminated 為 True 時發生)
        # 注意: 不同的 Scenario 對 collision 的定義可能不同，這裡檢查 info 中的標記
        # 如果是訓練早期，可以加重這個懲罰
        is_collision = False
        if 'wall_collision' in info and info['wall_collision']:
            is_collision = True
        elif 'collision' in info and info['collision']:  # 相容性檢查
            is_collision = True

        if is_collision:
            reward += self.collision_penalty

        # --- 3. 穩定性懲罰 (Stability Penalty) ---
        # 懲罰動作的劇烈變化: -weight * ||action - last_action||^2
        # 我們希望方向盤 (action[0]) 和 油門 (action[1]) 的變化是平滑的
        if self.last_action is not None:
            # 假設 action 是 numpy array
            action_diff = action - self.last_action
            # 計算 L2 Norm 的平方作為懲罰項
            stability_loss = np.sum(np.square(action_diff))
            reward -= self.stability_weight * stability_loss

        # 更新 last_action
        self.last_action = np.copy(action)

        # --- 4. 生存/時間懲罰 (Survival Reward) ---
        # 為了避免車子停在原地不動 (如果進度獎勵不夠大)，可以給予微小的生存獎勵
        # 或者給予微小的時間懲罰 (Time Penalty) 鼓勵盡快完成
        # 這裡我們採用生存獎勵，鼓勵它不要因為害怕扣分而立刻撞牆結束
        reward += self.survival_reward

        return obs, reward, terminated, truncated, info