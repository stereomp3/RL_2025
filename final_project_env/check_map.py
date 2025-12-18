import numpy as np
import os

# 設定你要檢查的目標地圖
TARGET_MAP = 'circle_ccw_competition'
base_dir = 'models/scenes'
starts_path = os.path.join(base_dir, TARGET_MAP, 'maps', 'starts.npz')

print(f"🧐 正在檢查檔案: {os.path.abspath(starts_path)}")

if not os.path.exists(starts_path):
    print("❌ 錯誤：找不到檔案！請確認資料夾名稱是否正確。")
    exit()

try:
    data = np.load(starts_path, allow_pickle=True)
    print(f"✅ 檔案讀取成功。包含 Keys: {list(data.keys())}")

    found_any = False
    for key in data:
        poses = data[key]
        print(f"\n--- Key: {key} ---")

        # 統一轉成 2D 陣列方便處理
        if poses.ndim == 1:
            poses = [poses]

        for i, pose in enumerate(poses):
            # 檢查最後一個數值 (Yaw)
            if len(pose) == 6:  # Euler [x, y, z, r, p, yaw]
                yaw = pose[5]
                fmt = "Euler (6)"
            elif len(pose) == 7:  # Quaternion
                # 簡單轉換一下看 Yaw
                import pybullet as p

                orn = pose[3:]
                euler = p.getEulerFromQuaternion(orn)
                yaw = euler[2]
                fmt = "Quaternion (7)"
            else:
                yaw = 0
                fmt = f"Unknown ({len(pose)})"

            print(f"  起始點 {i}: 位置={np.round(pose[:3], 2)}, Yaw={yaw:.4f} ({np.degrees(yaw):.1f}°), 格式={fmt}")

            # 判定結果
            if abs(abs(yaw) - 3.14) < 0.5:
                print("    🎉 成功！這個起始點是反向的 (180度)。")
            elif abs(yaw) < 0.5:
                print("    ❌ 失敗！這個起始點還是正向的 (0度)。")
            else:
                print("    ❓ 未知角度")

except Exception as e:
    print(f"讀取發生錯誤: {e}")