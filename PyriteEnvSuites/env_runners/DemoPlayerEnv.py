import json
import glob
import cv2
import numpy as np
from typing import Dict, Any
from pathlib import Path


class DemoReplayEnv:
    """
    Replay environment that reads from raw demo data:
    - robot_data_0.json: robot poses and timestamps
    - wrench_data_0.json: force/torque and timestamps  
    - rgb_0/*.jpg: camera images
    """
    
    def __init__(self, demo_store_path: str, query_sizes: Dict[str, int]):
        self.demo_path = Path(demo_store_path)
        self.current_hardware_time_s = 0.0
        self._idx = 0
        self._qs = query_sizes
        
        # Load robot data (fix trailing commas in JSON)
        with open(self.demo_path / "robot_data_0.json", "r") as f:
            content = f.read()
            # Remove trailing commas before } or ]
            content = content.replace(",\n\t}", "\n\t}").replace(",\n]", "\n]")
            self.robot_data = json.loads(content)
        
        # Load wrench data (fix trailing commas in JSON)
        with open(self.demo_path / "wrench_data_0.json", "r") as f:
            content = f.read()
            # Remove trailing commas before } or ]
            content = content.replace(",\n\t}", "\n\t}").replace(",\n]", "\n]")
            self.wrench_data = json.loads(content)
        
        # Get sorted list of image files
        img_pattern = str(self.demo_path / "rgb_0" / "*.jpg")
        self.img_files = sorted(glob.glob(img_pattern))
        
        # Use robot data length as episode length
        self._N = len(self.robot_data)
        
        print(f"Loaded episode: {self._N} robot frames, {len(self.wrench_data)} wrench frames, {len(self.img_files)} images")

    def reset(self) -> None:
        self._idx = 0
        self.current_hardware_time_s = self.robot_data[0]["robot_time_stamps"] / 1000.0

    def get_observation_from_buffer(self) -> Dict[str, Any]:
        """Get observation with sliding window matching ManipServerEnv interface"""
        
        def slice_last(data_list, q: int):
            start = max(0, self._idx - q + 1)
            end = self._idx + 1
            return data_list[start:end]
        
        raw: Dict[str, Any] = {}
        
        # Robot poses (ts_pose_fb_0 is 7D: x,y,z,quat)
        robot_window = slice_last(self.robot_data, self._qs["ts_pose_fb"])
        ts_poses = np.array([r["ts_pose_fb"] for r in robot_window])
        raw["ts_pose_fb_0"] = ts_poses
        raw["robot0_eef_pos"] = ts_poses  # alias for compatibility
        
        # Timestamps in seconds
        robot_timestamps = np.array([r["robot_time_stamps"] / 1000.0 for r in robot_window])
        raw["robot_time_stamps_0"] = robot_timestamps
        raw["rgb_time_stamps_0"] = robot_timestamps  # use same timestamps for rgb
        
        # Wrench data - find closest matches by timestamp
        current_time_ms = self.robot_data[self._idx]["robot_time_stamps"]
        wrench_window = []
        wrench_timestamps = []
        for i in range(max(0, self._idx - self._qs["wrench"] + 1), self._idx + 1):
            target_time = self.robot_data[i]["robot_time_stamps"]
            # Find closest wrench sample
            closest_idx = min(range(len(self.wrench_data)), 
                            key=lambda j: abs(self.wrench_data[j]["wrench_time_stamps"] - target_time))
            wrench_window.append(self.wrench_data[closest_idx]["wrench"])
            wrench_timestamps.append(self.wrench_data[closest_idx]["wrench_time_stamps"] / 1000.0)
        
        raw["wrench_0"] = np.array(wrench_window)
        raw["robot0_eef_wrench"] = np.array(wrench_window)  # alias
        raw["wrench_time_stamps_0"] = np.array(wrench_timestamps)
        
        # RGB images - match by timestamp from filename
        rgb_window = []
        for i in range(max(0, self._idx - self._qs["rgb"] + 1), self._idx + 1):
            target_time = self.robot_data[i]["robot_time_stamps"]
            # Find closest image by parsing timestamp from filename
            # Format: img_000445_45211.201273_ms.jpg -> extract 45211.201273
            def extract_timestamp(path):
                try:
                    # Split by underscore and get the timestamp part (second-to-last)
                    parts = path.split("_")
                    return float(parts[-2])
                except (ValueError, IndexError):
                    return 0.0
            
            closest_img = min(self.img_files,
                            key=lambda path: abs(extract_timestamp(path) - target_time))
            img = cv2.imread(closest_img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_window.append(img_rgb)
        
        raw["rgb_0"] = np.array(rgb_window)
        
        return raw

    def step_time(self) -> bool:
        """Advance to next timestep"""
        if self._idx < self._N - 1:
            self._idx += 1
            self.current_hardware_time_s = self.robot_data[self._idx]["robot_time_stamps"] / 1000.0
            return True
        return False
