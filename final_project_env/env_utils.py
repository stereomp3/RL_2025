import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

# 引用 racecar_gym 資料夾中的 RaceEnv
from racecar_gym.env import RaceEnv


# =============================================================================
# 自定義 Wrapper: 影像預處理 (128x128 RGB -> 64x64 Gray)
# =============================================================================
class ImageProcessWrapper(gym.ObservationWrapper):
    def __init__(self, env, resize_dim=(64, 64)):
        super().__init__(env)
        self.resize_dim = resize_dim
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(resize_dim[0], resize_dim[1], 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        # 1. RaceEnv 輸出格式為 (C, H, W) -> (3, 128, 128)
        # OpenCV 需要 (H, W, C) -> (128, 128, 3)
        img = np.transpose(obs, (1, 2, 0))

        # 2. Resize (128x128 -> 64x64)
        img = cv2.resize(img, (self.resize_dim[1], self.resize_dim[0]), interpolation=cv2.INTER_AREA)

        # 3. 轉灰階 (64, 64)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 4. 增加通道維度 (64, 64, 1)
        img = np.expand_dims(img, -1)

        return img


# =============================================================================
# 環境建立工廠
# =============================================================================

try:
    from reward_utils import RacecarRewardWrapper
except ImportError:
    RacecarRewardWrapper = None


def make_racecar_env(scenario, rank, seed=0, render_mode='rgb_array_birds_eye', use_shaped_reward=False):
    def _init():
        # 1. 使用 RaceEnv 建立基礎環境
        # RaceEnv action_space Box(2,)
        env = RaceEnv(
            scenario=scenario,
            render_mode=render_mode,
            reset_when_collision=True
        )

        env.reset(seed=seed + rank)

        # 2. (可選) 獎勵塑形
        if use_shaped_reward and RacecarRewardWrapper:
            env = RacecarRewardWrapper(
                env,
            )

        # 3. 影像處理
        env = ImageProcessWrapper(env, resize_dim=(64, 64))

        return env

    return _init


def get_vectorized_env(scenario_names, n_envs=4, seed=42, use_shaped_reward=False):
    env_fns = []

    if isinstance(scenario_names, str):
        scenario_names = [scenario_names]

    for i in range(n_envs):
        scenario = scenario_names[i % len(scenario_names)]

        env_fns.append(make_racecar_env(
            scenario=scenario,
            rank=i,
            seed=seed,
            use_shaped_reward=use_shaped_reward
        ))

    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    return vec_env