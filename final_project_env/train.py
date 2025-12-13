import argparse
import os
import time
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# 引用我們先前寫好的模組
from env_utils import get_vectorized_env
from models import POLICY_KWARGS


def train(algo, steps, shaped_reward, seed, experiment_name):
    # --- 1. 定義多地圖訓練集 ---
    # 這裡我們混合 Circle 和 Austria 進行訓練
    # 這樣模型既能學會 Circle 的高速過彎，也能學會 Austria 的複雜路況
    training_scenarios = [
        "scenarios/circle_cw.yml",
        "scenarios/austria.yml"
        # 如果你有更多地圖，例如 "scenarios/barcelona.yml"，都可以加進來
    ]
    # 修改實驗名稱以反映這是混合訓練
    reward_type = "Shaped" if shaped_reward else "Standard"
    run_name = f"{algo}_MultiMap_{reward_type}_{experiment_name}"
    save_dir = f"./logs/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"========================================")
    print(f"開始多地圖混合訓練")
    print(f"訓練地圖列表: {training_scenarios}")
    print(f"========================================")

    # 2. 建立多地圖並行環境
    env = get_vectorized_env(
        scenario_paths=training_scenarios,  # 傳入列表
        n_envs=4,
        seed=seed,
        use_shaped_reward=shaped_reward
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
    # parser.add_argument("--map", type=str, default="circle_cw", help="Scenario .yml file name")
    parser.add_argument("--steps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--shaped_reward", action="store_true", help="Use shaped reward (Method 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp_name", type=str, default="exp1", help="Custom experiment tag")

    args = parser.parse_args()

    train(args.algo, args.steps, args.shaped_reward, args.seed, args.exp_name)