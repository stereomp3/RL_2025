import argparse
import time
import os
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# 引用你現有的環境設置工具
from env_utils import make_racecar_env
from models import POLICY_KWARGS  # 如果載入時需要指定架構 (通常 load 會自動讀取，但保持引用是好習慣)


def enjoy(algo, map_name, model_path):
    # 1. 設定地圖路徑
    scenario_path = f"scenarios/{map_name}.yml"

    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"錯誤: 找不到模型檔案 {model_path}")
        return

    print(f"========================================")
    print(f"正在載入模型: {model_path}")
    print(f"測試地圖: {map_name}")
    print(f"========================================")

    # 2. 建立測試環境 (重點：render_mode='human')
    # 我們使用 DummyVecEnv 來包裝單一環境，這樣它的格式會跟訓練時的 Vectorized Env 一樣
    env_fn = make_racecar_env(
        scenario_path=scenario_path,
        rank=0,
        seed=42,
        render_mode='human',  # <--- 開啟視窗渲染模式
        use_shaped_reward=False  # 測試時不需要獎勵塑形，我們只看它跑得好不好
    )

    # 建立環境
    env = DummyVecEnv([env_fn])

    # 3. 套用跟訓練時完全一樣的堆疊與轉換
    # 這是非常重要的，否則模型會看到錯誤的輸入維度
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # 4. 載入模型
    if algo == "PPO":
        # custom_objects 用於解決 Python 版本差異可能導致的序列化問題
        model = PPO.load(model_path, env=env)
    elif algo == "A2C":
        model = A2C.load(model_path, env=env)
    else:
        raise ValueError("Algo must be PPO or A2C")

    # 5. 開始跑回圈
    obs = env.reset()
    print("\n開始駕駛! 按 Ctrl+C 可以停止程式...")

    try:
        while True:
            # deterministic=True 代表模型會選擇「它認為最好」的動作 (不隨機探索)
            action, _states = model.predict(obs, deterministic=True)

            # 執行動作
            obs, rewards, dones, info = env.step(action)

            # (可選) 稍微降速，讓肉眼看得比較清楚，不想等待可以註解掉
            # time.sleep(0.01)

            if dones[0]:
                print(f"回合結束 (撞牆或完成圈數) - 重置環境")
                # SB3 的 VecEnv 會自動 Reset，我們不需要手動呼叫 env.reset()

    except KeyboardInterrupt:
        print("停止測試。")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "A2C"], help="使用的算法")
    # austria, circle_cw, circle_cw_competition_collisionStop
    parser.add_argument("--map", type=str, default="circle_cw", help="要測試的地圖名稱")
    parser.add_argument("--model", type=str, required=True, help="模型路徑 (不需加 .zip)")

    args = parser.parse_args()

    enjoy(args.algo, args.map, args.model)