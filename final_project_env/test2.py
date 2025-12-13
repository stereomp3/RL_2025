import argparse
import time
import os
import cv2  # 用來顯示圖片
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from env_utils import make_racecar_env


def enjoy_visual(algo, map_name, model_path, view_mode):
    # 1. 設定地圖與視角
    scenario_path = f"scenarios/{map_name}.yml"

    # 選擇渲染模式
    # 'rgb_array_birds_eye': 上帝視角 (適合看整體跑法)
    # 'rgb_array_follow':    跟隨視角 (像賽車遊戲)
    render_mode = f"rgb_array_{view_mode}"

    print(f"========================================")
    print(f"載入模型: {model_path}")
    print(f"地圖: {map_name} | 視角: {view_mode}")
    print(f"========================================")

    # 2. 建立環境
    # 注意: 這裡我們不使用 'human'，而是使用 rgb_array 模式讓環境回傳圖片
    env_fn = make_racecar_env(
        scenario_path=scenario_path,
        rank=0,
        seed=42,
        render_mode=render_mode,  # <--- 關鍵修改
        use_shaped_reward=False
    )

    env = DummyVecEnv([env_fn])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # 3. 載入模型
    if algo == "PPO":
        model = PPO.load(model_path, env=env)
    elif algo == "A2C":
        model = A2C.load(model_path, env=env)

    # 4. 開始執行
    obs = env.reset()
    print("\n按 'q' 鍵離開視窗...")

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)

            # --- 視覺化核心代碼 ---
            # 從環境獲取渲染圖片
            img = env.render()

            # 確保圖片格式正確 (有時候可能是 List 或 Batch 結構)
            if isinstance(img, list):
                img = img[0]

            # 轉換顏色 (RGB -> BGR) 這是 OpenCV 的標準
            if img is not None:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # 在圖片上顯示文字 (可選)
                cv2.putText(img_bgr, f"Map: {map_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 顯示視窗
                cv2.imshow("Racecar Visualizer", img_bgr)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # --------------------

            if dones[0]:
                print("回合結束，重置環境")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "A2C"])
    parser.add_argument("--map", type=str, default="circle_cw")
    parser.add_argument("--model", type=str, required=True, help="模型路徑")
    # 新增視角選擇參數
    parser.add_argument("--view", type=str, default="birds_eye", choices=["birds_eye", "follow"],
                        help="birds_eye (上帝視角) 或 follow (跟隨視角)")

    args = parser.parse_args()

    enjoy_visual(args.algo, args.map, args.model, args.view)