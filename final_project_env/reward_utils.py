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


class RacecarRewardWrapper2(gym.Wrapper):
    """
    激進型競速獎勵 (Aggressive Racing Reward)
    目標: 極速、推進、不計代價
    Progress Reward (進度獎勵)：係數提升到 2000.0。這是最重要的指標，告訴車子「向前跑」的權重是最高的。
    Motor Reward (油門獎勵)：將權重提升，鼓勵「地板油」。如果 action[0] 是油門，我們給予正向獎勵。
    Velocity Reward (新增)：除了油門，我們直接獎勵「真實速度」。如果車子雖然踩油門但卡住，這項就不會給分。
    Bang-Bang Penalty (穩定性懲罰)：係數降低到 0.05。為了極限過彎，允許車手做出比較劇烈的修正，不要因為怕扣分而不敢轉方向盤。
    """

    def __init__(self, env, stability_weight=0.05, collision_penalty=-100.0):
        super().__init__(env)
        self.stability_weight = stability_weight  # 降低平滑要求 (0.1 -> 0.05)
        self.collision_penalty = collision_penalty

        # 記錄變數
        self.last_action = None
        self.last_total_progress = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = np.zeros(self.action_space.shape)
        self.last_total_progress = info.get('lap', 0) + info.get('progress', 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- 1. Motor Reward (油門獎勵) ---
        # 鼓勵油門踩到底 (範圍 -1 ~ 1)
        # 增加權重到 2.0，強烈暗示它 "Go!"
        motor_reward = 2.0 * float(action[0])

        # --- 2. Velocity Reward (真實速度獎勵 - 新增) ---
        # 如果 info 中有速度資訊，直接獎勵速度大小
        # 這比單純獎勵油門更準確
        velocity_reward = 0.0
        if 'velocity' in info:
            # velocity 通常是 (x, y, z) 向量
            speed = np.linalg.norm(info['velocity'])
            velocity_reward = 0.1 * speed  # 根據速度給予額外獎勵

        # --- 3. Bang-Bang Penalty (動作平滑度) ---
        # 降低權重，允許高難度操作
        action_diff = np.abs(action - self.last_action)
        bang_bang_penalty = -self.stability_weight * np.sum(action_diff)

        # --- 4. Progress Reward (進度獎勵) ---
        current_total_progress = info.get('lap', 0) + info.get('progress', 0)
        progress_delta = current_total_progress - self.last_total_progress

        if progress_delta < -0.5:
            progress_delta = 0.0

        # 超級加倍，讓「推進」成為絕對核心目標
        progress_reward = 2000.0 * progress_delta

        # --- 5. Wall Collision Penalty (撞牆懲罰) ---
        collision_reward = 0.0
        if info.get('wall_collision', False):
            collision_reward = self.collision_penalty  # -100
            terminated = True  # 撞牆直接結束，不浪費時間

        # --- 計算總分 ---
        new_reward = motor_reward + velocity_reward + bang_bang_penalty + progress_reward + collision_reward

        # 更新狀態
        self.last_action = np.copy(action)
        self.last_total_progress = current_total_progress

        return obs, new_reward, terminated, truncated, info


class RacecarRewardWrapper3(gym.Wrapper):
    """
    修正版激進型獎勵 (Smart Aggressive Racing Reward)
    目標: 依然追求極速，但學會「慢進快出」的過彎技巧
         加入 Stuck Penalty (怠速懲罰): 防止車子裝死
    """

    def __init__(self, env, stability_weight=0.05, collision_penalty=-500.0, stuck_penalty=-0.5):
        super().__init__(env)
        self.stability_weight = stability_weight
        self.collision_penalty = collision_penalty  # 加重撞牆懲罰 (-100 -> -500)
        self.stuck_penalty = stuck_penalty  # 停在原地會扣分

        self.last_action = None
        self.last_total_progress = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = np.zeros(self.action_space.shape)
        self.last_total_progress = info.get('lap', 0) + info.get('progress', 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 解析動作
        motor = float(action[0])  # 油門 (-1 ~ 1)
        steering = float(action[1])  # 轉向 (-1 ~ 1)

        # 獲取速度資訊
        speed = 0.0
        if 'velocity' in info:
            speed = np.linalg.norm(info['velocity'])

        # --- 1. Dynamic Motor Reward (動態油門獎勵) ---
        # 直線時 (steering 小)，鼓勵大油門
        # 彎道時 (steering 大)，抑制油門獎勵，甚至懲罰全油門過彎
        steering_magnitude = abs(steering)

        if steering_magnitude < 0.3:
            # 直線衝刺: 全力獎勵油門
            motor_reward = 2.0 * motor
        else:
            # 彎道: 減少油門獎勵，鼓勵適當收油
            # 如果轉向很大還全油門，獎勵會變得很小甚至負的
            motor_reward = 0.5 * motor

        # --- 2. High-Speed Steering Penalty (高速轉向懲罰 - 新增) ---
        # 懲罰 "速度快 + 大轉向" 的危險行為
        # 這會教會模型: 想轉彎? 先減速!
        # 係數 0.2 可以根據訓練狀況調整
        danger_penalty = -0.2 * (speed * steering_magnitude)

        # --- 3. Velocity Reward (真實速度獎勵) ---
        velocity_reward = 0.1 * speed

        # --- 4. Bang-Bang Penalty (動作平滑度) ---
        action_diff = np.abs(action - self.last_action)
        bang_bang_penalty = -self.stability_weight * np.sum(action_diff)

        # --- 5. Progress Reward (進度獎勵 - 核心) ---
        current_total_progress = info.get('lap', 0) + info.get('progress', 0)
        progress_delta = current_total_progress - self.last_total_progress

        if progress_delta < -0.5:
            progress_delta = 0.0

        progress_reward = 2000.0 * progress_delta

        # --- 6. Wall Collision Penalty (撞牆懲罰 - 加重) ---
        collision_reward = 0.0
        if info.get('wall_collision', False):
            collision_reward = self.collision_penalty  # -500
            terminated = True

        # --- 7. Stuck Penalty (停止懲罰) ---
        # 如果速度極低，視為卡住或裝死
        stuck_reward = 0.0
        if speed < 0.05:
            stuck_reward = self.stuck_penalty  # 速度太低，減分

        # --- 計算總分 ---
        new_reward = (motor_reward +
                      velocity_reward +
                      bang_bang_penalty +
                      progress_reward +
                      collision_reward +
                      danger_penalty +
                      stuck_reward)

        # 更新狀態
        self.last_action = np.copy(action)
        self.last_total_progress = current_total_progress

        return obs, new_reward, terminated, truncated, info
