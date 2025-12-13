import argparse
import os
import time
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# 引用我們先前寫好的模組
from env_utils import get_vectorized_env
from models import POLICY_KWARGS


def train(algo, map_name, steps, shaped_reward, seed, experiment_name):
    # 1. 設定路徑與參數
    # 根據你的資料夾結構調整 scenarios 路徑
    scenario_path = f"scenarios/{map_name}.yml"

    # 定義實驗名稱 (用於 Tensorboard 和 存檔)
    reward_type = "Shaped" if shaped_reward else "Standard"
    run_name = f"{algo}_{map_name}_{reward_type}_{experiment_name}"
    save_dir = f"./logs/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"========================================")
    print(f"開始訓練: {run_name}")
    print(f"算法: {algo} | 地圖: {map_name}")
    print(f"獎勵機制: {reward_type}")
    print(f"總步數: {steps}")
    print(f"========================================")

    # 2. 建立並行環境 (Vectorized Environment)
    # CPU 核心數越多，n_envs 可以設越大 (建議 4-8)
    env = get_vectorized_env(
        scenario_path=scenario_path,
        n_envs=4,
        seed=seed,
        use_shaped_reward=shaped_reward  # 切換獎勵函數
    )

    # 3. 初始化模型
    # 根據參數選擇 PPO 或 A2C
    if algo == "PPO":
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=POLICY_KWARGS,  # 使用自定義 CNN
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            learning_rate=3e-4,
            n_steps=2048,  # PPO 每次更新採樣的步數
            batch_size=64,
            n_epochs=10,
            ent_coef=0.01,  # 熵係數，增加探索
        )
    elif algo == "A2C":
        model = A2C(
            "CnnPolicy",
            env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            learning_rate=7e-4,
            n_steps=20,  # A2C 更新頻率較高
            ent_coef=0.01,
        )
    else:
        raise ValueError("Algorithm must be PPO or A2C")

    # 4. 設定回調函數 (Callbacks)
    # 每 50,000 步儲存一次模型檢查點
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="model_ckpt"
    )

    # 5. 開始訓練
    start_time = time.time()
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=run_name
    )

    # 6. 儲存最終模型
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    env.close()

    print(f"訓練完成! 耗時: {(time.time() - start_time) / 60:.2f} 分鐘")
    print(f"模型已儲存至: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "A2C"], help="RL Algorithm")
    parser.add_argument("--map", type=str, default="circle_cw", help="Scenario .yml file name")
    parser.add_argument("--steps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--shaped_reward", action="store_true", help="Use shaped reward (Method 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp_name", type=str, default="exp1", help="Custom experiment tag")

    args = parser.parse_args()

    train(args.algo, args.map, args.steps, args.shaped_reward, args.seed, args.exp_name)