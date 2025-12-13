import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from tqdm import tqdm
from PIL import Image  # 需要安裝 pillow: pip install pillow

# 引用你現有的環境設置工具
from env_utils import make_racecar_env


def save_gif(algo, map_name, model_path, output_filename="result.gif"):
    # 1. 設定地圖路徑
    scenario_path = f"scenarios/{map_name}.yml"

    print(f"========================================")
    print(f"載入模型: {model_path}")
    print(f"測試地圖: {map_name}")
    print(f"輸出檔案: {output_filename}")
    print(f"========================================")

    # 2. 建立環境
    # 使用 'rgb_array_birds_eye' 以獲得上帝視角的彩色圖片
    env_fn = make_racecar_env(
        scenario_path=scenario_path,
        rank=0,
        seed=42,
        render_mode='rgb_array_birds_eye',
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

    # 4. 開始錄製
    img_list = []
    obs = env.reset()
    done = False

    # 設定最大步數，避免跑太久
    max_steps = 1000
    print(f"開始錄製... (最大步數: {max_steps})")

    for i in tqdm(range(max_steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # 獲取圖片
        img = env.render()

        if isinstance(img, list):
            img = img[0]

        # --- 新增檢查 ---
        if img is not None:
            img_list.append(img)
        else:
            # 如果發現 None，印出警告 (只印一次避免洗版)
            if i == 0:
                print(f"警告: 第 {i} 步 env.render() 回傳了 None，無法錄製該幀。請檢查 render_mode 設定。")
        # ---------------

        if done[0]:
            print("回合結束!")
            break

    env.close()

    # 5. 儲存成 GIF
    if len(img_list) > 0:
        print(f"正在儲存 GIF ({len(img_list)} 幀)...")
        # 將 numpy arrays 轉換為 PIL Images
        pil_imgs = [Image.fromarray(img) for img in img_list]

        # 儲存
        pil_imgs[0].save(
            output_filename,
            save_all=True,
            append_images=pil_imgs[1:],
            duration=40,  # 每幀持續時間 (ms), 40ms = 25fps
            loop=0  # 0 = 無限循環
        )
        print(f"完成! 檔案已儲存為: {output_filename}")
    else:
        print("錯誤: 沒有擷取到任何畫面。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "A2C"])
    parser.add_argument("--map", type=str, default="circle_cw")
    parser.add_argument("--model", type=str, required=True, help="模型路徑")
    parser.add_argument("--output", type=str, default="result.gif", help="輸出的 GIF 檔名")

    args = parser.parse_args()

    save_gif(args.algo, args.map, args.model, args.output)