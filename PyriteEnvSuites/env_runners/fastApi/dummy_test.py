# dummy_test.py
import numpy as np
import requests

SERVER = "http://192.168.1.141:8000"  # adjust if needed

def main():
    # Raw sequence lengths required by the MPC (from your config)
    # rgb horizon = 2, down_sample = 10  -> len_rgb = 11
    # low horizon = 3, down_sample = 5  -> len_low = 11
    # wrench horizon = 7000, down_sample = 1 -> len_wrench = 7000
    len_rgb = 11
    len_low = 11
    len_wrench = 7000

    # Shapes from shape_meta.obs:
    # rgb_0: [3, 224, 224]  (channels=3, H=224, W=224)
    # For dummy obs_task we assume channels-last: [T, H, W, C]
    rgb_0 = np.zeros((len_rgb, 224, 224, 3), dtype=np.float32)

    # Low-dim:
    # robot0_eef_pos: [T_low, 3]
    # robot0_eef_rot_axis_angle: [T_low, 6]
    # robot0_eef_wrench: [T_wrench, 6]
    robot0_eef_pos = np.zeros((len_low, 3), dtype=np.float32)
    robot0_eef_rot = np.zeros((len_low, 6), dtype=np.float32)
    robot0_eef_wrench = np.zeros((len_wrench, 6), dtype=np.float32)

    # Timestamps (1D, no batch)
    t = np.linspace(0.0, 1.0, num=len_wrench, dtype=np.float32)

    obs_task = {
        "rgb_0": rgb_0.tolist(),
        "robot0_eef_pos": robot0_eef_pos.tolist(),
        "robot0_eef_rot_axis_angle": robot0_eef_rot.tolist(),
        "robot0_eef_wrench": robot0_eef_wrench.tolist(),
        "rgb_time_stamps_0": t.tolist(),
        "robot_time_stamps_0": t.tolist(),
        "wrench_time_stamps_0": t.tolist(),
    }

    print("Sending /set_observation ...")
    r = requests.post(f"{SERVER}/set_observation", json={"obs_task": obs_task})
    print("set_observation:", r.status_code)
    print("set_observation body:", r.text)

    print("Sending /compute_sparse_control ...")
    r2 = requests.post(f"{SERVER}/compute_sparse_control", json={"time_now": 1.0})
    print("compute_sparse_control:", r2.status_code)
    print("compute_sparse_control body:", r2.text)
    try:
        print("response JSON keys:", r2.json().keys())
    except Exception as e:
        print("Failed to parse JSON:", e)

if __name__ == "__main__":
    main()
