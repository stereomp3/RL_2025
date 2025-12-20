import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==============================
# 設定區
# ==============================
LOG_DIR = "./tensorboard_logs/"  # 你的 TensorBoard Log 路徑
Metric_Tag = "rollout/ep_len_mean"  # SB3 預設的獎勵標籤 (平均每回合獎勵)
# 其他可用標籤: 'train/loss', 'train/value_loss', 'rollout/ep_len_mean'

SMOOTHING = 0.9  # 平滑係數 (0~1)，越大越平滑，0 代表不平滑
FIGURE_SIZE = (12, 6)


# ==============================

def smooth_curve(values, weight=0.6):
    """
    使用指數移動平均 (EMA) 來平滑曲線，讓趨勢更明顯
    """
    last = values[0]
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def parse_tensorboard(path):
    """
    讀取單個 tfevents 檔案並轉為 DataFrame
    """
    # 初始化 EventAccumulator
    event_acc = EventAccumulator(path)
    event_acc.Reload()

    # 檢查標籤是否存在
    if Metric_Tag not in event_acc.Tags()['scalars']:
        print(f"⚠️ 警告: 在 {path} 中找不到標籤 '{Metric_Tag}'，跳過。")
        return None

    # 提取數據 (Step, Value)
    scalars = event_acc.Scalars(Metric_Tag)
    steps = [x.step for x in scalars]
    values = [x.value for x in scalars]

    df = pd.DataFrame({'Step': steps, 'Value': values})
    return df


def main():
    print(f"🔍 正在搜尋 {LOG_DIR} 下的實驗紀錄...")

    # 搜尋所有的 events.out.tfevents 檔案 (遞迴搜尋)
    # 結構通常是: tensorboard_logs/實驗名稱_1/events.out.tfevents...
    log_files = glob.glob(os.path.join(LOG_DIR, "**", "events.out.tfevents*"), recursive=True)

    all_data = []

    for log_file in log_files:
        # 取得實驗名稱 (通常是資料夾名稱)
        # 例如: ./tensorboard_logs/PPO_circle_cw_1/events... -> PPO_circle_cw_1
        dir_name = os.path.dirname(log_file)
        exp_name = os.path.basename(dir_name)

        print(f"   -> 正在讀取: {exp_name}")
        df = parse_tensorboard(log_file)

        if df is not None and not df.empty:
            # 進行平滑處理
            df['Smoothed_Value'] = smooth_curve(df['Value'], weight=SMOOTHING)
            df['Experiment'] = exp_name
            all_data.append(df)

    if not all_data:
        print("❌ 沒有找到任何有效的訓練數據！")
        return

    # 合併所有數據
    full_df = pd.concat(all_data, ignore_index=True)

    # ==============================
    # 開始繪圖
    # ==============================
    print("📈 正在繪製圖表...")
    plt.figure(figsize=FIGURE_SIZE)
    sns.set_theme(style="darkgrid")

    # 使用 Seaborn 繪圖，它會自動處理顏色跟圖例
    # x軸: Step (訓練步數), y軸: Smoothed_Value (平滑後的獎勵)
    sns.lineplot(
        data=full_df,
        x="Step",
        y="Smoothed_Value",
        hue="Experiment",  # 根據實驗名稱分顏色
        linewidth=2.0
    )

    plt.title(f"Model Training Progress ({Metric_Tag})", fontsize=16)
    plt.xlabel("Timesteps (Training Steps)", fontsize=12)
    plt.ylabel("Average Episode Reward (Smoothed)", fontsize=12)
    plt.legend(title="Experiment Name", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 自動調整佈局避免被切掉
    plt.tight_layout()

    # 儲存與顯示
    save_path = "training_comparison.png"
    # plt.savefig(save_path, dpi=300)
    print(f"✅ 圖表已儲存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()