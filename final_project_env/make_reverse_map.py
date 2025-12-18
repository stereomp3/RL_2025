import numpy as np
import os
import shutil
import pybullet as p


def create_true_reverse_map(source_scene_name, target_scene_name):
    # --- 路徑設定 ---
    base_dir = 'models/scenes'
    src_dir = os.path.join(base_dir, source_scene_name)
    dst_dir = os.path.join(base_dir, target_scene_name)

    print(f"=== 開始製作反向地圖: {target_scene_name} ===")

    # 1. 複製整個資料夾 (如果已存在則先刪除)
    if os.path.exists(dst_dir):
        print(f"刪除舊的 {dst_dir}...")
        shutil.rmtree(dst_dir)

    print(f"複製資料夾: {src_dir} -> {dst_dir}")
    shutil.copytree(src_dir, dst_dir)

    # 2. 檔案更名 (.yml 和 .sdf)
    old_yml = os.path.join(dst_dir, f"{source_scene_name}.yml")
    new_yml = os.path.join(dst_dir, f"{target_scene_name}.yml")

    old_sdf = os.path.join(dst_dir, f"{source_scene_name}.sdf")
    new_sdf = os.path.join(dst_dir, f"{target_scene_name}.sdf")

    if os.path.exists(old_yml):
        os.rename(old_yml, new_yml)
        # 修改 YML 內容
        with open(new_yml, 'r', encoding='utf-8') as f:
            content = f.read()
        # 取代名稱與 sdf 引用
        content = content.replace(source_scene_name, target_scene_name)
        with open(new_yml, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已更新設定檔: {new_yml}")

    if os.path.exists(old_sdf):
        os.rename(old_sdf, new_sdf)
        print(f"已更新模型檔: {new_sdf}")

    # 3. 修改 starts.npz (旋轉車頭 180 度)
    starts_path = os.path.join(dst_dir, 'maps', 'starts.npz')
    if os.path.exists(starts_path):
        print("正在反轉起始點 (starts.npz)...")
        data = np.load(starts_path, allow_pickle=True)
        new_starts = {}

        for key in data:
            poses = data[key]
            # 處理單一或多個 pose
            if poses.ndim == 1:
                poses = [poses]
                is_single = True
            else:
                is_single = False

            new_pose_list = []
            for pose in poses:
                pose = np.array(pose, dtype=float)
                # 判斷格式是 Euler(6) 還是 Quaternion(7)
                if len(pose) == 6:
                    pos = pose[:3]
                    euler = pose[3:]
                    euler[2] += np.pi  # Yaw + 180度
                    new_pose_list.append(np.concatenate([pos, euler]))
                elif len(pose) == 7:
                    pos = pose[:3]
                    orn = pose[3:]
                    euler = p.getEulerFromQuaternion(orn)
                    new_yaw = euler[2] + np.pi
                    new_orn = p.getQuaternionFromEuler([euler[0], euler[1], new_yaw])
                    new_pose_list.append(np.concatenate([pos, new_orn]))

            val = np.array(new_pose_list)
            if is_single:
                val = val[0]
            new_starts[key] = val

        np.savez(starts_path, **new_starts)
        print("起始點反轉完成。")

    # 4. ★★★ 關鍵步驟：修改 maps.npz (反轉路徑順序) ★★★
    # 這一步讓 "前進" 變成真正的 "前進"，不需要負分獎勵
    maps_path = os.path.join(dst_dir, 'maps', 'maps.npz')
    if os.path.exists(maps_path):
        print("正在反轉路徑點 (maps.npz)...")
        map_data = np.load(maps_path, allow_pickle=True)
        new_map_data = {}

        # maps.npz 通常包含 'centerline', 'inner_border', 'outer_border' 等
        for key in map_data:
            arr = map_data[key]
            # 我們只反轉那些看起來像是一連串座標的陣列 (Nx2 或 Nx3)
            # 這是為了讓 Checkpoints 的順序變成逆時針
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 10:
                print(f"  -> 反轉陣列: {key} (Shape: {arr.shape})")
                new_map_data[key] = arr[::-1]  # 反轉陣列順序
            else:
                new_map_data[key] = arr

        np.savez(maps_path, **new_map_data)
        print("路徑邏輯反轉完成！")

    print(f"\n=== 成功！地圖 {target_scene_name} 已建立 ===")
    print("現在你可以用正常的 Reward (progress_reward: 1.0) 來訓練右轉了！")


if __name__ == "__main__":
    # 使用範例: 把 circle_cw 變成真正的 circle_ccw
    # 注意: 來源資料夾名稱要正確
    create_true_reverse_map('circle_cw_competition', 'circle_ccw_competition')