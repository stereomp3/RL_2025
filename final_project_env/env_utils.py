import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import racecar_gym.envs.gym_api  # 註冊 racecar 環境
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from reward_utils import RacecarRewardWrapper
import cv2  # pip install opencv-python


class RacecarActionWrapper(gym.ActionWrapper):
    """
    將 SB3 輸出的連續向量動作 [steering, motor]
    轉換為 RacecarGym 需要的字典格式 {'steering': ..., 'motor': ...}
    """

    def __init__(self, env):
        super().__init__(env)
        # 定義新的動作空間：2維連續向量 [-1, 1]
        # index 0: steering (轉向)
        # index 1: motor (油門/煞車)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def action(self, action):
        # 將向量拆解並包裝成字典
        # 注意: 原始環境要求的值通常是 shape (1,) 的 array
        return {
            'steering': np.array([action[0]], dtype=np.float32),
            'motor': np.array([action[1]], dtype=np.float32)
        }


class RacecarImageWrapper(gym.ObservationWrapper):
    """
    提取影像並調整大小，保持 RGB 格式
    """

    def __init__(self, env, resize_dim=(64, 64)):
        super().__init__(env)
        self.resize_dim = resize_dim
        # 定義 Observation Space 為 (64, 64, 3) -> RGB
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(resize_dim[0], resize_dim[1], 3),
            dtype=np.uint8
        )

    def observation(self, obs):
        target_key = None
        candidates = ['video', 'camera_front', 'rgb_camera', 'images', 'camera']

        for key in candidates:
            if key in obs:
                target_key = key
                break

        if target_key is None:
            raise ValueError(f"RacecarImageWrapper 找不到影像數據! 環境回傳的鍵值有: {list(obs.keys())}")

        image = obs[target_key]
        image = np.array(image, dtype=np.uint8)

        # Resize
        if image.shape[0] != self.resize_dim[0] or image.shape[1] != self.resize_dim[1]:
            image = cv2.resize(image, (self.resize_dim[1], self.resize_dim[0]), interpolation=cv2.INTER_AREA)

        return image


class GrayScaleObservation(gym.ObservationWrapper):
    """
    將 (H, W, 3) RGB 影像轉為 (H, W, 1) 灰階
    """

    def __init__(self, env, keep_dim=True):
        super().__init__(env)
        self.keep_dim = keep_dim
        # 寬鬆檢查，只要是 3 通道即可
        if len(env.observation_space.shape) == 3 and env.observation_space.shape[-1] == 3:
            obs_shape = env.observation_space.shape[:2]
            if self.keep_dim:
                self.observation_space = spaces.Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1),
                                                    dtype=np.uint8)
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # 防禦性程式設計：確保輸入真的是 RGB
        if observation.ndim == 3 and observation.shape[-1] == 3:
            gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            if self.keep_dim:
                return np.expand_dims(gray, -1)
            return gray
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        # 繼承原本的通道數 (可能是 1 或 3)
        channels = self.observation_space.shape[2] if len(self.observation_space.shape) > 2 else 1
        self.observation_space = spaces.Box(low=0, high=255, shape=self.shape + (channels,), dtype=np.uint8)

    def observation(self, observation):
        # 執行 Resize
        obs = cv2.resize(observation, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)

        # 如果原本有通道維度 (例如 (64, 64, 1))，但 cv2.resize 輸出了 (64, 64)，必須把通道加回來
        if len(self.observation_space.shape) == 3 and obs.ndim == 2:
            obs = np.expand_dims(obs, -1)
        # ----------------

        return obs
# ---------------------------------------------------------
# 環境工廠函數 (Environment Factory)
# ---------------------------------------------------------
# render_mode: ['human', 'rgb_array_follow', 'rgb_array_birds_eye', 'rgb_array_lidar']
def make_racecar_env(scenario_path, rank, seed=0, render_mode="rgb_array_birds_eye", use_shaped_reward=False):
    """
    用於 SubprocVecEnv 的工具函數。
    它會建立環境並套用一系列的 Wrappers。

    Args:
        scenario_path: .yml 場景設定檔的路徑 (Circle, Austria 等)
        rank: 環境的索引 (用於設定隨機種子)
        seed: 全域隨機種子
        render_mode: 是否渲染 ('human' 或 'rgb_array')
        use_shaped_reward (bool): 是否啟用自定義的獎勵塑形 (True=Method 2, False=Method 1)
    """
    mode = render_mode if render_mode is not None else 'rgb_array_birds_eye'
    def _init():
        # 1. 建立原始環境 並傳入 scenario 設定
        env = gym.make(
            'SingleAgentRaceEnv-v0',
            scenario=scenario_path,
            render_mode=mode
        )
        env.reset(seed=seed + rank)  # 設定隨機種子，確保實驗可重現
        # 必須最早套用，將 Dict 動作轉為 Vector，讓後面的 Agent 和 Wrapper 都能看懂
        env = RacecarActionWrapper(env)
        # 2. (可選) 獎勵塑形
        if use_shaped_reward and RacecarRewardWrapper:
            env = RacecarRewardWrapper(
                env,
                stability_weight=0.5,  # 0.5 表示中等程度的平滑要求
                collision_penalty=-5.0,  # 撞牆扣 5 分
                survival_reward=0.01  # 暫時設為 0.01，如果為 0 代表專注於跑得快
            )

        # 3. 提取影像並 Resize 到 64x64 (RGB)
        # 注意: 這裡已經先 Resize 成 64x64x3 了
        env = RacecarImageWrapper(env, resize_dim=(64, 64))

        # 4. 轉灰階 (64, 64, 3) -> (64, 64, 1)
        # 現在這裡不會報錯了，因為輸入確是 3 通道
        env = GrayScaleObservation(env, keep_dim=True)

        # 5. 二次 Resize (保險用，實際上這步現在是多餘但無害的)
        env = ResizeObservation(env, (64, 64))

        return env

    return _init


# ---------------------------------------------------------
# 建立並行化環境的主函數
# ---------------------------------------------------------

def get_vectorized_env(scenario_paths, n_envs=4, seed=42, use_shaped_reward=False):
    """
    建立支援多地圖混合訓練的 Vectorized Environment。

    Args:
        scenario_paths (list): 地圖路徑的列表，例如 ['scenarios/circle_cw.yml', 'scenarios/austria.yml']
        n_envs: 並行環境數量
    """
    env_fns = []

    # 確保 scenario_paths 是一個列表
    if isinstance(scenario_paths, str):
        scenario_paths = [scenario_paths]

    for i in range(n_envs):
        # --- 關鍵邏輯: 輪流分配地圖 ---
        # 如果有 2 張圖，4 個環境：
        # env_0 -> map[0] (Circle)
        # env_1 -> map[1] (Austria)
        # env_2 -> map[0] (Circle)
        # env_3 -> map[1] (Austria)
        scenario_path = scenario_paths[i % len(scenario_paths)]

        env_fns.append(make_racecar_env(scenario_path, i, seed, use_shaped_reward=use_shaped_reward))

    # 使用 SubprocVecEnv 讓每個環境在獨立的 Process 跑 (加速關鍵!)
    vec_env = SubprocVecEnv(env_fns)

    # 堆疊幀數 (Frame Stacking) - 這是關鍵!
    # 將連續 4 幀疊在一起，變成 (4, 64, 64)
    # 這樣神經網路才能「看到」速度和加速度
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # 轉換 Image 通道 (H, W, C) -> (C, H, W)
    # PyTorch 需要 Channel First 格式
    vec_env = VecTransposeImage(vec_env)

    return vec_env

# ---------------------------------------------------------
# 測試代碼 (使用者檢查用)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 測試路徑 (請根據你實際的路徑修改)
    # 例如: 'scenarios/circle_cw.yml'
    test_scenario = 'scenarios/circle_cw.yml'

    if os.path.exists(test_scenario):
        print(f"正在建立 {test_scenario} 的向量化環境...")

        # 建立 2 個並行環境作為測試
        env = get_vectorized_env(test_scenario, n_envs=2)

        print("環境建立成功!")
        print(f"Observation Space: {env.observation_space.shape}")
        # 預期輸出: (2, 4, 64, 64) -> (Batch, Stacked_Frames, H, W)

        obs = env.reset()
        print(f"Reset Output Shape: {obs.shape}")

        action = np.array([env.action_space.sample() for _ in range(2)])
        obs, rewards, dones, infos = env.step(action)
        print(f"Step Reward Shape: {rewards.shape}")

        env.close()
        print("測試完成。")
    else:
        print(f"找不到路徑: {test_scenario}，請檢查你的資料夾結構。")
