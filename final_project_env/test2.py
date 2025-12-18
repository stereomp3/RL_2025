import os
import sys

# ==========================================
# ★★★ 強制設定地圖路徑 (Fix Path Issue) ★★★
# ==========================================
# 取得當前專案的絕對路徑
current_dir = os.path.abspath(os.path.dirname(__file__))

# 1. 將 models 資料夾的路徑設為環境變數
# racecar_gym 會優先讀取這個變數
models_path = os.path.join(current_dir, 'models')
os.environ['RACECAR_ASSETS_DIR'] = models_path

# 2. 確保 Python 優先引用本地的 racecar_gym (如果有的話)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"🌍 強制指定地圖資源路徑: {models_path}")
# ==========================================
import time
import cv2
import numpy as np
import argparse
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

# 1. 引用 Server 端使用的環境 Wrapper
from racecar_gym.env import RaceEnv


# ==========================================
# Agent: 模擬 Client 端的模型與預處理
# ==========================================
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


# ==========================================
# Dashboard: 模擬 Server 端的畫面合成
# ==========================================
def render_dashboard(env, obs, info):
    """
    合成上帝視角、跟隨視角與數據的儀表板
    """
    progress = info.get('progress', 0)
    lap = int(info.get('lap', 0))
    score = lap + progress - 1.

    # 取得底層環境 (繞過 Wrapper)
    base_env = env.env.unwrapped
    agent_id = base_env._scenario.agent.id
    world = base_env._scenario.world

    # --- 強制渲染多視角 (解決 force_render 缺失問題) ---
    def safe_render(mode, **kwargs):
        try:
            # 嘗試直接呼叫 world.render，這是最穩定的方法
            real_mode = mode.replace('rgb_array_', '')
            return world.render(mode=real_mode, agent_id=agent_id, **kwargs)
        except Exception as e:
            print(f"渲染錯誤 ({mode}): {e}")
            return np.zeros((kwargs.get('height', 100), kwargs.get('width', 100), 3), dtype=np.uint8)

    img1 = safe_render('rgb_array_higher_birds_eye', width=540, height=540, position=np.array([4.89, -9.30, -3.42]),
                       fov=120)
    img2 = safe_render('rgb_array_birds_eye', width=270, height=270)
    img3 = safe_render('rgb_array_follow', width=128, height=128)

    # 觀測圖 (需轉置)
    img4 = np.transpose(obs, (1, 2, 0)).astype(np.uint8)

    # 畫布合成
    canvas = np.zeros((540, 810, 3), dtype=np.uint8)
    if img1 is not None: canvas[0:540, 0:540, :] = img1
    if img2 is not None: canvas[:270, 540:810, :] = img2
    if img3 is not None: canvas[270 + 10:270 + 128 + 10, 540 + 7:540 + 128 + 7, :] = img3
    # 貼上模型看到的畫面
    canvas[270 + 10:270 + 128 + 10, 540 + 128 + 14:540 + 128 + 128 + 14, :] = cv2.resize(img4, (128, 128))

    # 文字繪製
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype('./racecar_gym/Arial.ttf', 25)
        font_large = ImageFont.truetype('./racecar_gym/Arial.ttf', 35)
    except:
        font = ImageFont.load_default()
        font_large = ImageFont.load_default()

    draw.text((550, 450), f"Score {score:.3f}", font=font_large, fill=(255, 255, 255))
    draw.text((688, 280), "Obs", font=font, fill=(255, 87, 34))

    return np.asarray(pil_img)


# ==========================================
# 主程式
# ==========================================
def main():
    # --- 設定區 ---
    # MODEL_PATH = "./logs/PPO_circle_cw_competition_collisionStop_austria_competition_Shaped_transformer_reward/model_ckpt_200000_steps"  # 請修改為你的模型路徑
    MODEL_PATH = "./logs/final/PPO3_76000000_steps.zip"  # 請修改為你的模型路徑
    # SCENARIO = "austria_competition"  # (circle_cw_competition_collisionStop, austria_competition)
    SCENARIO = "circle_ccw_competition_collisionStop"  # (circle_cw_competition_collisionStop, austria_competition)
    # -------------

    print(f"=== 本地整合測試 (Scenario: {SCENARIO}) ===")

    # 1. 建立環境 (使用 RaceEnv)
    try:
        env = RaceEnv(
            scenario=SCENARIO,
            render_mode='rgb_array_birds_eye',
            reset_when_collision=True
        )

        print("環境建立成功!")
    except Exception as e:
        print(f"環境建立失敗: {e}")
        return

    # 2. 載入模型
    try:
        agent = LocalAgent(MODEL_PATH)
    except Exception as e:
        print(f"模型載入失敗: {e}")
        env.close()
        return

    # 3. 測試迴圈
    obs, info = env.reset()
    print("\n開始跑分! 按 'q' 鍵離開...")
    # ==========================================
    # ★★★ 修正版 Debug 程式碼 (v3) ★★★
    # ==========================================
    print("\n" + "=" * 40)
    print("🔍 DEBUG 資訊檢查 (v3)")
    print("=" * 40)
    try:
        # 取得底層環境與場景
        base_env = env.env.unwrapped
        scenario = getattr(base_env, '_scenario', None)

        # 1. 檢查載入的地圖名稱
        if scenario:
            world_name = scenario.world._config.name
            print(f"1. 目前載入的地圖 (World Name): 【 {world_name} 】")
        else:
            print("無法取得 Scenario 物件")

        # 2. 取得車輛 ID
        vehicle_id = None
        if hasattr(scenario, 'agents') and 'A' in scenario.agents:
            # 多人模式
            # 修正點: 使用 ._id 或 .id
            vehicle_id = scenario.agents['A'].vehicle._id
        elif hasattr(scenario, 'agent'):
            # 單人模式
            # 修正點: 使用 ._id 或 .id
            vehicle_id = scenario.agent.vehicle._id

        # 3. 檢查實際位置與角度
        if vehicle_id is not None:
            import pybullet
            pos, orn = pybullet.getBasePositionAndOrientation(vehicle_id)
            euler = pybullet.getEulerFromQuaternion(orn)

            yaw_degree = np.degrees(euler[2])
            print(f"2. 車輛位置 (x, y, z): {pos}")
            print(f"3. 車輛角度 (Yaw): {euler[2]:.4f} (約 {yaw_degree:.1f} 度)")

            # 判斷是否反向 (Yaw 接近 3.14 或 -3.14 代表轉了 180 度)
            if abs(abs(euler[2]) - 3.14) < 0.5:
                print("   ✅ 狀態: 車頭已朝向反向 (逆時針) - 成功！")
            elif abs(euler[2]) < 0.5:
                print("   ❌ 狀態: 車頭朝向正向 (順時針) - 起始點未改變")
                print("      -> 請確認您是否重新執行過 make_reverse_map.py")
                print("      -> 請確認 starts.npz 是否覆蓋了正確的資料夾")
            else:
                print(f"   ❓ 狀態: 車頭朝向其他角度 ({yaw_degree:.1f} 度)")
        else:
            print("無法取得 Vehicle ID (vehicle._id)")

    except Exception as e:
        print(f"Debug 過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    print("=" * 40 + "\n")
    # ==========================================
    try:
        while True:
            # Agent 決策
            action = agent.get_action(obs)

            # 環境互動
            obs, reward, terminal, truncated, info = env.step(action)

            # 渲染儀表板
            dashboard_img = render_dashboard(env, obs, info)

            # 顯示
            bgr_img = cv2.cvtColor(dashboard_img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Local Test Dashboard", bgr_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if terminal or truncated:
                print(f"回合結束! 最終進度: {info.get('progress', 0):.3f}")
                obs, info = env.reset()
                agent.frame_buffer.clear()  # 重置 Frame Stack

    except KeyboardInterrupt:
        print("測試中止")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
