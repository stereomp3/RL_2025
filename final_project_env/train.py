import argparse
import os
import time
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback

# 引用更新後的 env_utils
from env_utils import get_vectorized_env
from models import POLICY_KWARGS
import sys
from functools import wraps
from datetime import datetime


# 自動儲存 log
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def tee_log(log_file=None):
    """裝飾器：將 print 輸出同時存檔與顯示"""
    if log_file is None:
        log_file = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            with open(log_file, "w", encoding="utf-8") as f:
                sys.stdout = Tee(original_stdout, f)
                try:
                    result = func(*args, **kwargs)
                finally:
                    sys.stdout = original_stdout
            print(f"✅ 輸出已保存到 {log_file}")
            return result

        return wrapper

    return decorator


def train(algo, map_names, steps, shaped_reward, seed, experiment_name):
    log_file = f"{experiment_name}.txt"
    original_stdout = sys.stdout
    with open(log_file, "w", encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)
        # 解析地圖名稱列表
        maps = map_names.split(',')  # 支援傳入 "circle_cw,austria"

        reward_type = "Shaped" if shaped_reward else "Standard"
        run_name = f"{algo}_{'_'.join(maps)}_{reward_type}_{experiment_name}"
        save_dir = f"./logs/{run_name}"
        os.makedirs(save_dir, exist_ok=True)

        print(f"========================================")
        print(f"開始訓練: {run_name}")
        print(f"地圖列表: {maps}")
        print(f"========================================")

        # 1. 建立環境
        env = get_vectorized_env(
            scenario_names=maps,  # 傳入名稱列表
            n_envs=80,
            seed=seed,
            use_shaped_reward=shaped_reward
        )

        # 2. 初始化模型
        if algo == "PPO":
            model = PPO(
                "CnnPolicy",
                env,
                policy_kwargs=POLICY_KWARGS,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                learning_rate=1e-4,
                n_steps=4096,
                batch_size=256,
                n_epochs=10,  # 每次收集完數據後，重複訓練幾次 # 保持 0.01 鼓勵初期探索
                ent_coef=0.01,  # 保持 0.01 鼓勵初期探索，
                max_grad_norm=0.5  # 防止 Transformer 梯度爆炸
            )
        elif algo == "A2C":
            model = A2C(
                "CnnPolicy",
                env,
                policy_kwargs=POLICY_KWARGS,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                learning_rate=1e-4,
                n_steps=20,
                ent_coef=0.01,
                max_grad_norm=0.5  # 防止 Transformer 梯度爆炸
            )

        # 3. 訓練
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=save_dir, name_prefix="model_ckpt")

        model.learn(total_timesteps=steps, callback=checkpoint_callback, tb_log_name=run_name, progress_bar=True)

        model.save(os.path.join(save_dir, "final_model"))
        env.close()
        print("訓練完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "A2C"])
    # 這裡改成傳入地圖名稱，用逗號分隔
    parser.add_argument("--maps", type=str, default="circle_cw",
                        help="Map names separated by comma (e.g. circle_cw,austria)")
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--shaped_reward", action="store_true") # 目前 reward 使用 3，原本為 2
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="exp1")

    args = parser.parse_args()

    train(args.algo, args.maps, args.steps, args.shaped_reward, args.seed, args.exp_name)
