import argparse
import json
import numpy as np
import requests
import cv2  # 需要 pip install opencv-python
from collections import deque
from stable_baselines3 import PPO


# 定義你的 Agent
class PPOAgent:
    class LocalAgent:
        def __init__(self, model_path):
            print(f"正在載入模型: {model_path}...")
            self.model = PPO.load(model_path)

            # Frame Stacking 緩衝區 (模擬 VecFrameStack)
            self.n_stack = 8
            self.frame_buffer = deque(maxlen=self.n_stack)

        def preprocess_image(self, obs):
            """
            處理 RaceEnv 回傳的畫面 (3, 128, 128) -> 模型需要的 (64, 64, 1)
            """
            # 1. RaceEnv 回傳的是 Channel-First (C, H, W)，轉為 OpenCv 的 (H, W, C)
            # obs shape: (3, 128, 128) -> img shape: (128, 128, 3)
            img = np.transpose(obs, (1, 2, 0))

            # 2. Resize 到 64x64
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

            # 3. 轉灰階
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # 4. 增加通道維度 (64, 64) -> (64, 64, 1)
            img = img[:, :, np.newaxis]

            return img

        def get_action(self, obs):
            # 1. 影像預處理
            processed_frame = self.preprocess_image(obs)

            # 2. 堆疊 Frames
            if len(self.frame_buffer) == 0:
                # 第一幀，重複填充
                for _ in range(self.n_stack):
                    self.frame_buffer.append(processed_frame)
            else:
                # 後續幀，推入佇列
                self.frame_buffer.append(processed_frame)

            # 3. 組合堆疊: (64, 64, 1) * 4 -> (64, 64, 4)
            stacked_obs = np.concatenate(list(self.frame_buffer), axis=2)

            # 4. 轉回 Channel-First 給 PyTorch: (64, 64, 4) -> (4, 64, 64)
            stacked_obs = np.transpose(stacked_obs, (2, 0, 1))

            # 5. 增加 Batch 維度: (4, 64, 64) -> (1, 4, 64, 64)
            stacked_obs_batch = np.expand_dims(stacked_obs, axis=0)

            # 6. 模型推論
            action, _ = self.model.predict(stacked_obs_batch, deterministic=True)

            return action[0]  # 取出 Batch 中的第一個動作
    def __init__(self, model_path):
        # 載入你訓練好的模型
        print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)

        # --- Frame Stacking Buffer ---
        # 用來儲存最近 8 幀的緩衝區
        self.n_stack = 8
        self.frame_buffer = deque(maxlen=self.n_stack)

    def preprocess_image(self, obs):
        """
        將 Server 傳來的 (3, 128, 128) 彩色圖
        轉換為模型需要的 (64, 64, 1) 灰階圖
        """
        # 1. 轉換為 (H, W, C) 以便 OpenCV 處理
        # Server 傳來的是 (C, H, W) -> (3, 128, 128)
        # 我們轉成 (128, 128, 3)
        img = np.transpose(obs, (1, 2, 0))

        # 2. Resize 到 64x64
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        # 3. 轉灰階 (64, 64)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 4. 增加通道維度變 (64, 64, 1)
        img = img[:, :, np.newaxis]

        return img

    def get_stacked_observation(self, obs):
        """
        處理 Frame Stacking
        """
        # 先進行單張圖片的預處理
        processed_frame = self.preprocess_image(obs)

        # 如果是第一幀 (Buffer 是空的)，將同一幀複製 8 次填滿
        if len(self.frame_buffer) == 0:
            for _ in range(self.n_stack):
                self.frame_buffer.append(processed_frame)
        else:
            # 否則推入新的一幀，最舊的會自動被擠掉
            self.frame_buffer.append(processed_frame)

        # 堆疊 frames: (64, 64, 1) * 4 -> (64, 64, 8)
        stacked_obs = np.concatenate(list(self.frame_buffer), axis=2)

        # 轉換為 Channel-First 給 PyTorch 模型: (8, 64, 64)
        stacked_obs = np.transpose(stacked_obs, (2, 0, 1))

        return stacked_obs

    def act(self, observation):
        # 1. 轉換原始資料格式
        obs = np.array(observation).astype(np.uint8)

        # 2. 獲取堆疊後的觀測值 (8, 64, 64)
        final_obs = self.get_stacked_observation(obs)

        # 3. 增加 Batch 維度 (SB3 需要 Batch Size) -> (1, 8, 64, 64)
        # 這樣就符合 ValueError 要求的 (n_env, 8, 64, 64) 格式了
        final_obs_batch = np.expand_dims(final_obs, axis=0)

        # 4. 預測動作
        action, _states = self.model.predict(final_obs_batch, deterministic=True)

        return action[0]  # 因為 batch size 是 1，所以取出第一個結果


def connect(agent, url: str = 'http://localhost:5000'):
    print(f"Connecting to server at {url}...")
    while True:
        # 1. 向 Server 拿畫面
        try:
            response = requests.get(f'{url}')
            data = json.loads(response.text)
        except Exception as e:
            print(f"Connection error: {e}")
            break

        if data.get('error'):
            print(f"Server Error: {data['error']}")
            break

        if data.get('terminal'):
            print('Episode finished.')
            return

        obs = data['observation']

        # 2. 決定動作 (透過 Agent 處理)
        action_to_take = agent.act(obs)

        # 3. 傳送動作回 Server
        try:
            response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
            result = json.loads(response.text)
            if result.get('terminal'):
                print('Episode finished.')
                return
        except Exception as e:
            print(f"Post Action Error: {e}")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000')
    # 預設模型路徑
    parser.add_argument('--model', type=str, default='./logs/final/PPO_continue_Austria_128000000_steps.zip')
    args = parser.parse_args()

    # 初始化你的 PPO Agent
    agent = PPOAgent(model_path=args.model)

    connect(agent, url=args.url)