import argparse
import os
import time
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

# 引用你的 env_utils 和 models
from env_utils import get_vectorized_env
from models import POLICY_KWARGS


def train(algo, map_names, steps, shaped_reward, seed, experiment_name, load_path=None):
    # 解析地圖名稱列表
    maps = map_names.split(',')

    reward_type = "Shaped" if shaped_reward else "Standard"

    # 如果是接續訓練，我們在實驗名稱後加上 "_continued" 以示區別 (可選)
    if load_path:
        experiment_name += "_continued"

    run_name = f"{algo}_{'_'.join(maps)}_{reward_type}_{experiment_name}"
    save_dir = f"./logs/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"========================================")
    if load_path:
        print(f"正在載入模型: {load_path}")
        print(f"接續訓練: {steps} 步")
    else:
        print(f"開始新訓練: {run_name}")
    print(f"地圖列表: {maps}")
    print(f"========================================")

    # 1. 建立環境
    env = get_vectorized_env(
        scenario_names=maps,
        n_envs=4,
        seed=seed,
        use_shaped_reward=shaped_reward
    )

    # 2. 初始化或載入模型
    if load_path and os.path.exists(load_path):
        # --- 接續訓練模式 ---
        if algo == "PPO":
            # 載入模型
            # env=env: 確保模型連接到當前的環境
            # tensorboard_log: 確保日誌繼續寫入
            model = PPO.load(load_path, env=env, tensorboard_log="./tensorboard_logs/")
        elif algo == "A2C":
            model = A2C.load(load_path, env=env, tensorboard_log="./tensorboard_logs/")

        print(">>> 模型載入成功! 準備繼續訓練...")
        reset_timesteps = False  # 關鍵: 告訴 TensorBoard 不要從 0 開始畫圖，而是接續之前的步數

    else:
        # --- 全新訓練模式 ---
        if load_path:
            print(f"警告: 找不到模型檔案 {load_path}，將建立新模型。")

        if algo == "PPO":
            model = PPO(
                "CnnPolicy",
                env,
                policy_kwargs=POLICY_KWARGS,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                learning_rate=1e-4,  # 保持你之前設定的大模型參數
                n_steps=4096,
                batch_size=256,
                n_epochs=10,
                ent_coef=0.01,
                max_grad_norm=0.5
            )
        elif algo == "A2C":
            model = A2C(
                "CnnPolicy",
                env,
                policy_kwargs=POLICY_KWARGS,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                learning_rate=7e-4,
                n_steps=20,
                ent_coef=0.01,
            )
        reset_timesteps = True

    # 3. 設定存檔回調
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="model_ckpt"
    )

    # 4. 開始學習
    # reset_num_timesteps=False 讓 log 延續
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=run_name,
        reset_num_timesteps=reset_timesteps
    )

    # 5. 存檔
    final_save_path = os.path.join(save_dir, "final_model_continued")
    model.save(final_save_path)
    env.close()
    print(f"訓練結束，模型已儲存至: {final_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "A2C"])
    parser.add_argument("--maps", type=str, default="circle_cw", help="Map names separated by comma")
    parser.add_argument("--steps", type=int, default=500000, help="要訓練多少步")
    parser.add_argument("--shaped_reward", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="exp1")
    # 如果要讀取舊的模型的話，可以使用 --load_model
    parser.add_argument("--load_model", type=str, default=None, help="舊模型路徑 (例如 logs/.../final_model.zip)")

    args = parser.parse_args()

    train(args.algo, args.maps, args.steps, args.shaped_reward, args.seed, args.exp_name, args.load_model)