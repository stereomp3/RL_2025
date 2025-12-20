import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 資料輸入區
# ==========================================
# 請將你的數據整理成以下的字典列表格式
# 你可以從你的 log 或 print 輸出中複製貼上並稍作整理
data = [
    # --- 模型 1 (A2C) 的數據 ---
    {"Steps": 80000000, "Mean Reward": 0.619677, "Std Reward": 0.046602, "Model": "PPO_01"},
    {"Steps": 104000000, "Mean Reward": 1.101951, "Std Reward": 0.788542, "Model": "PPO_01"},
    {"Steps": 128000000, "Mean Reward": 0.818334, "Std Reward": 0.212366, "Model": "PPO_01"},

    # --- 模型 2 (PPO) 的數據 ---
    # 這裡只是範例，請替換成你的真實數據
    {"Steps": 80000000, "Mean Reward": 0.45, "Std Reward": 0.10, "Model": "PPO_Baseline"},
    {"Steps": 104000000, "Mean Reward": 0.70, "Std Reward": 0.15, "Model": "PPO_Baseline"},
    {"Steps": 128000000, "Mean Reward": 0.95, "Std Reward": 0.12, "Model": "PPO_Baseline"},

    # --- 模型 3 (PPO2) ---
    {"Steps": 80000000, "Mean Reward": 0.20, "Std Reward": 0.05, "Model": "A2C_Test"},
    {"Steps": 104000000, "Mean Reward": 0.35, "Std Reward": 0.08, "Model": "A2C_Test"},
    {"Steps": 128000000, "Mean Reward": 0.40, "Std Reward": 0.10, "Model": "A2C_Test"},
]

# ==========================================
# 2. 處理數據
# ==========================================
df = pd.DataFrame(data)

# 確保數據按步數排序，以免畫線時亂跳
df = df.sort_values(by=["Model", "Steps"])

# ==========================================
# 3. 開始繪圖
# ==========================================
# 設定畫布大小與風格
plt.figure(figsize=(12, 7))
sns.set_theme(style="darkgrid")

# 取得所有唯一的模型名稱
models = df['Model'].unique()
# 產生對應數量的顏色
palette = sns.color_palette("tab10", n_colors=len(models))

# 迴圈畫出每一個模型的線條與陰影
for i, model_name in enumerate(models):
    # 篩選出該模型的數據
    model_data = df[df['Model'] == model_name]

    color = palette[i]

    # A. 畫主線 (平均分數)
    plt.plot(
        model_data["Steps"],
        model_data["Mean Reward"],
        marker='o',  # 數據點標記
        linestyle='-',  # 實線
        linewidth=2.5,  # 線寬
        label=model_name,  # 圖例名稱
        color=color
    )

    # B. 畫陰影 (標準差範圍: Mean - Std 到 Mean + Std)
    plt.fill_between(
        model_data["Steps"],
        model_data["Mean Reward"] - model_data["Std Reward"],
        model_data["Mean Reward"] + model_data["Std Reward"],
        alpha=0.2,  # 透明度 (0~1)
        color=color
    )

# ==========================================
# 4. 圖表修飾
# ==========================================
plt.title("Model Performance Comparison", fontsize=18, fontweight='bold')
plt.xlabel("Training Steps", fontsize=14)
plt.ylabel("Average Episode Reward", fontsize=14)

# 設定圖例位置
plt.legend(title="Models", title_fontsize='13', fontsize='12', loc='upper left')

# 如果步數很大，使用科學記號或簡化顯示 (例如 100M)
# 這裡保持原樣，Matplotlib 通常會自動優化

# 調整佈局
plt.tight_layout()

# 儲存圖片
save_filename = "model_comparison_plot.png"
plt.savefig(save_filename, dpi=300)
print(f"✅ 圖表已建立並儲存為: {save_filename}")

# 顯示圖片
plt.show()