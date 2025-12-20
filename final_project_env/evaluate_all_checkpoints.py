import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
import gymnasium as gym
from racecar_gym.env import RaceEnv

# ==========================================
# 設定區
# ==========================================
# 你的模型存檔路徑
LOG_DIR = "./logs/test_A2C/"
# 測試用的地圖
SCENARIO = "austria_competition"
# 每個模型要測幾次
N_EVAL_EPISODES = 3
# 影像處理設定
RESIZE_DIM = (64, 64)


# ==========================================

# 1. 定義影像處理 Wrapper (必須與訓練時完全一樣)
class ImageProcessWrapper(gym.ObservationWrapper):
    def __init__(self, env, resize_dim=(64, 64)):
        super().__init__(env)
        self.resize_dim = resize_dim
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(resize_dim[0], resize_dim[1], 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        import cv2
        img = np.transpose(obs, (1, 2, 0))
        img = cv2.resize(img, (self.resize_dim[1], self.resize_dim[0]), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, -1)
        return img


def make_env():
    # 建立環境
    env = RaceEnv(
        scenario=SCENARIO,
        render_mode='rgb_array_birds_eye',
        reset_when_collision=True
    )
    env = ImageProcessWrapper(env)
    return env


def extract_steps(filename):
    """從檔名解析步數，例如 model_ckpt_600000_steps.zip -> 600000"""
    match = re.search(r'_(\d+)_steps', filename)
    if match:
        return int(match.group(1))
    return -1


def main():
    # 1. 搜尋所有模型檔案
    # 支援兩種命名格式: "PPO_t.zip" (最終版) 或 "model_ckpt_123_steps.zip"
    files = glob.glob(os.path.join(LOG_DIR, "*.zip"))

    if not files:
        print(f"❌ 在 {LOG_DIR} 找不到任何 .zip 檔案")
        return

    # 建立測試環境
    print("🛠️ 正在建立測試環境...")
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=8)  # 注意: 這裡要跟訓練時的 n_stack 一樣 (4 或 8)
    env = VecTransposeImage(env)

    results = []

    print(f"🔍 找到 {len(files)} 個模型，開始評估...")

    # 排序檔案 (按步數)
    files.sort(key=extract_steps)

    for model_path in files:
        steps = extract_steps(os.path.basename(model_path))

        # 如果解析不出步數 (例如 final_model.zip)，就設為最大值或忽略
        if steps == -1:
            if "final" in model_path:
                steps = 999999999  # 視為最後
            else:
                continue

        print(f"   -> 正在測試模型: {os.path.basename(model_path)} (Steps: {steps})")

        try:
            # 載入模型
            model = PPO.load(model_path, env=env)

            # 進行評估
            mean_reward, std_reward = evaluate_policy(
                model,
                env,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True
            )

            print(f"      分數: {mean_reward:.2f} +/- {std_reward:.2f}")

            results.append({
                "Steps": steps,
                "Mean Reward": mean_reward,
                "Std Reward": std_reward,
                "Model": os.path.basename(model_path)
            })

        except Exception as e:
            print(f"      ❌ 載入失敗: {e}")

    env.close()

    if not results:
        print("沒有成功評估任何模型。")
        return

    # 2. 轉為 DataFrame 並繪圖
    df = pd.DataFrame(results)
    # 過濾掉極端值 (如果需要)
    # df = df[df['Steps'] < 100000000]

    print("\n📊 評估結果:")
    print(df.sort_values(by="Mean Reward", ascending=False).head())

    # 繪圖
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="darkgrid")

    # 畫主線
    sns.lineplot(data=df, x="Steps", y="Mean Reward", marker="o", linewidth=2.5)

    # 畫標準差陰影
    plt.fill_between(
        df["Steps"],
        df["Mean Reward"] - df["Std Reward"],
        df["Mean Reward"] + df["Std Reward"],
        alpha=0.2
    )

    plt.title(f"Model Performance vs Training Steps ({SCENARIO})", fontsize=16)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Average Episode Reward", fontsize=12)

    # 標示最高分點
    best_row = df.loc[df['Mean Reward'].idxmax()]
    plt.annotate(
        f'Best: {best_row["Mean Reward"]:.1f}',
        xy=(best_row['Steps'], best_row['Mean Reward']),
        xytext=(best_row['Steps'], best_row['Mean Reward'] + 50),
        arrowprops=dict(facecolor='red', shrink=0.05),
    )

    save_path = "checkpoint_evaluation.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 圖表已儲存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()