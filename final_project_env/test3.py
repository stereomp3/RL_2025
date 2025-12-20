import numpy as np
import gymnasium as gym
import cv2
from PIL import Image
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# 引用 RaceEnv
from racecar_gym.env import RaceEnv


# ==========================================
# 必要: 影像預處理 Wrapper (需與訓練時一致)
# ==========================================
class ImageProcessWrapper(gym.ObservationWrapper):
    def __init__(self, env, resize_dim=(64, 64)):
        super().__init__(env)
        self.resize_dim = resize_dim
        # 定義為 (64, 64, 1) 的灰階圖 (Channel Last)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(resize_dim[0], resize_dim[1], 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        # 1. RaceEnv 預設輸出為 (C, H, W) -> (3, 128, 128)
        # 轉為 OpenCV 需要的 (H, W, C) -> (128, 128, 3)
        img = np.transpose(obs, (1, 2, 0))

        # 2. Resize
        img = cv2.resize(img, (self.resize_dim[1], self.resize_dim[0]), interpolation=cv2.INTER_AREA)

        # 3. 轉灰階
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 4. 增加通道維度 -> (64, 64, 1)
        img = np.expand_dims(img, -1)
        return img


# ==========================================
# 主程式
# ==========================================
def main():
    # --- 設定區 ---
    # SCENARIO = 'circle_cw_competition_collisionStop'
    SCENARIO = 'austria_competition'
    # SCENARIO = 'barcelona'
    CHECKPOINT_DIR = './logs/final/'
    MODEL_NAME = 'PPO_continue_Austria_128000000_steps.zip'  # 不需加 .zip

    # -------------

    # 1. 建立並包裝環境
    def make_env():
        env = RaceEnv(
            scenario=SCENARIO,
            render_mode='rgb_array_birds_eye',
            reset_when_collision=True
        )
        env = ImageProcessWrapper(env)  # 套用預處理
        return env

    # 使用 DummyVecEnv 包裝 (SB3 格式)
    env = DummyVecEnv([make_env])
    # 堆疊 8 幀 (還原訓練設定)
    env = VecFrameStack(env, n_stack=8)
    # 轉為 PyTorch 格式 (Batch, Channel, Height, Width)
    env = VecTransposeImage(env)

    # 2. 載入模型
    print(f"Loading model from {CHECKPOINT_DIR}{MODEL_NAME}...")
    model = PPO.load(CHECKPOINT_DIR + MODEL_NAME, env=env)

    # 3. 執行測試迴圈
    img_list = []
    obs = env.reset()
    done = False

    # 設定最大步數
    progress_bar = tqdm(range(5000))
    total_reward = np.zeros(1)

    for i in progress_bar:
        # 模型推論
        action, lstm_states = model.predict(obs, deterministic=True)

        # 環境互動
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

        # --- 擷取畫面 ---
        # 注意: obs 的形狀是 (Batch, Channel, Height, Width) -> (1, 4, 64, 64)
        # 我們取最後一幀 (最新的畫面)
        frame = obs[0][-1, :, :]  # 取出 (64, 64) 的灰階圖
        img_list.append(frame)

        # 更新進度條資訊
        # info[0] 是因為 VecEnv 回傳的是列表
        progress = info[0].get('progress', 0)
        progress_bar.set_description(f'action: {action[0]}, progress: {progress:.5f}')

        if done[0]:
            break

    env.close()

    # 4. 儲存 GIF
    if len(img_list) > 0:
        print("Saving result.gif...")
        imgs = [Image.fromarray(img) for img in img_list]
        imgs[0].save("result.gif", save_all=True, append_images=imgs[1:], duration=10, loop=0)
        print("Done!")
    else:
        print("No frames captured.")


if __name__ == "__main__":
    main()
