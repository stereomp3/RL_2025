import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from stable_baselines3 import PPO

# 引用 Server 端使用的環境
from racecar_gym.env import RaceEnv


# ==========================================
# 1. 模擬 Client 端的模型與預處理邏輯
# ==========================================
class LocalAgent:
    def __init__(self, model_path):
        print(f"載入模型: {model_path}...")
        self.model = PPO.load(model_path)
        self.n_stack = 4
        self.frame_buffer = deque(maxlen=self.n_stack)

    def preprocess_image(self, obs):
        """Client 端: 將 (3, 128, 128) 轉為 (64, 64, 1)"""
        # (C, H, W) -> (H, W, C)
        img = np.transpose(obs, (1, 2, 0))
        # Resize & GrayScale
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Add channel dimension
        img = img[:, :, np.newaxis]
        return img

    def get_action(self, obs):
        """模擬 Client 接收畫面並產生動作"""
        # 1. 預處理單幀
        processed_frame = self.preprocess_image(obs)

        # 2. Frame Stacking
        if len(self.frame_buffer) == 0:
            for _ in range(self.n_stack):
                self.frame_buffer.append(processed_frame)
        else:
            self.frame_buffer.append(processed_frame)

        # (64, 64, 1)*4 -> (64, 64, 4)
        stacked_obs = np.concatenate(list(self.frame_buffer), axis=2)
        # (H, W, C) -> (C, H, W) 給 PyTorch
        stacked_obs = np.transpose(stacked_obs, (2, 0, 1))
        # 加 Batch 維度 -> (1, 4, 64, 64)
        stacked_obs_batch = np.expand_dims(stacked_obs, axis=0)

        # 3. 推論
        action, _ = self.model.predict(stacked_obs_batch, deterministic=True)
        return action[0]


# ==========================================
# 2. 模擬 Server 端的畫面合成邏輯 (Dashboard)
# ==========================================
def render_dashboard(env, obs, info, sid="TEST_USER"):
    """
    這段代碼直接複製自 server.py 的 get_img_views，
    用於驗證 force_render 是否會報錯。
    """
    progress = info['progress']
    lap = int(info['lap'])
    score = lap + progress - 1.

    # --- 關鍵測試點: 呼叫 force_render ---
    # 如果這裡報錯，代表 single_agent_race.py 還沒修好
    try:
        # 使用 .unwrapped 繞過 OrderEnforcing Wrapper (如果 server.py 沒改，這裡模擬修改後的效果)
        base_env = env.env.unwrapped

        img1 = base_env.force_render(render_mode='rgb_array_higher_birds_eye', width=540, height=540,
                                     position=np.array([4.89, -9.30, -3.42]), fov=120)
        img2 = base_env.force_render(render_mode='rgb_array_birds_eye', width=270, height=270)
        img3 = base_env.force_render(render_mode='rgb_array_follow', width=128, height=128)
    except AttributeError:
        print("錯誤: 找不到 force_render，請確保已修改 single_agent_race.py 並在 server.py 使用 .unwrapped")
        # 回傳假圖避免程式崩潰
        return np.zeros((540, 810, 3), dtype=np.uint8)

    img4 = (obs.transpose((1, 2, 0))).astype(np.uint8)

    # 組合圖片 (Server 邏輯)
    img = np.zeros((540, 810, 3), dtype=np.uint8)
    if img1 is not None: img[0:540, 0:540, :] = img1
    if img2 is not None: img[:270, 540:810, :] = img2
    if img3 is not None: img[270 + 10:270 + 128 + 10, 540 + 7:540 + 128 + 7, :] = img3
    img[270 + 10:270 + 128 + 10, 540 + 128 + 14:540 + 128 + 128 + 14, :] = img4

    # 繪製文字
    try:
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        # 嘗試載入字體，若無則使用預設
        try:
            font = ImageFont.truetype('./racecar_gym/Arial.ttf', 25)
            font_large = ImageFont.truetype('./racecar_gym/Arial.ttf', 35)
        except:
            font = ImageFont.load_default()
            font_large = ImageFont.load_default()

        draw.text((5, 5), "Full Map", font=font, fill=(255, 87, 34))
        draw.text((550, 10), "Bird's Eye", font=font, fill=(255, 87, 34))
        draw.text((550, 450), f"Score {score:.3f}", font=font_large, fill=(255, 255, 255))

        img = np.asarray(pil_img)
    except Exception as e:
        print(f"繪圖警告: {e}")

    return img


# ==========================================
# 3. 主程式: 整合測試
# ==========================================
def main():
    # 設定參數
    model_path = "./logs/PPO_circle_cw_Standard_baseline/final_model"  # 請確認路徑正確
    # scenario = "austria_competition"
    scenario = "circle_cw_competition_collisionStop"

    print("=== 開始本地整合測試 ===")

    # 1. 建立環境 (模擬 Server 啟動)
    # 注意: RaceEnv 來自 server.py 的 import，確保測試的是相同的環境包裝
    try:
        env = RaceEnv(scenario=scenario, render_mode='rgb_array_birds_eye', reset_when_collision=True)
    except Exception as e:
        print(f"環境建立失敗: {e}")
        print("提示: 請確認 scenarios/circle_cw.yml 是否存在且內容正確。")
        return

    # 2. 建立代理 (模擬 Client 連線)
    try:
        agent = LocalAgent(model_path)
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return

    obs, info = env.reset()
    print("\n測試開始! 按 'q' 離開...")

    try:
        while True:
            # --- Client 端邏輯 ---
            action = agent.get_action(obs)

            # --- 環境互動 ---
            obs, reward, terminal, truncated, info = env.step(action)

            # --- Server 端邏輯 (生成 Dashboard) ---
            dashboard_img = render_dashboard(env, obs, info)

            # --- 顯示結果 (本地視窗) ---
            # OpenCV 使用 BGR，PIL 使用 RGB，需轉換顏色
            bgr_img = cv2.cvtColor(dashboard_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Local Integration Test", bgr_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if terminal or truncated:
                print(f"回合結束! Score: {info['lap'] + info['progress'] - 1:.3f}")
                obs, info = env.reset()
                agent.frame_buffer.clear()  # 重置 Frame Stack

    except KeyboardInterrupt:
        print("停止測試。")
    finally:
        env.close()  # 注意: RaceEnv 沒有 close 方法可能會報錯，這裡只是習慣性呼叫
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()