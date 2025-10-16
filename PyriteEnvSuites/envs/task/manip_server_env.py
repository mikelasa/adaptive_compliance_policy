import os
import sys

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
PACKAGE_PATH = os.path.join(SCRIPT_PATH, "../../")
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../../"))

from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Type
import numpy as np
import time
from multiprocessing import Process, Lock
from multiprocessing.managers import SharedMemoryManager, BaseManager
from einops import rearrange


from PyriteUtility.planning_control.filtering import LiveLPFilter
from PyriteUtility.computer_vision.computer_vision_utility import get_image_transform
from PyriteUtility.data_pipeline.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)
from PyriteUtility.data_pipeline.shared_memory.shared_memory_util import (
    ArraySpec,
)

from hardware_interfaces.workcell.table_top_manip.python import (
    manip_server_pybind as ms,
)

input_res = (224, 224)


class ManipServerEnv:
    """
    A class to interact with the real robot for the vase wiping task.

    """

    def __init__(
        self,
        camera_res_hw: List[int],
        hardware_config_path: str,
        filter_params: dict,
        query_sizes: Dict[str, int],
        compliant_dimensionality: int,
    ) -> None:
        # manipulation server
        # pybind links to the C++ code of the manipulation server and allows using its functions in C++
        server = ms.ManipServer()
        if not server.initialize(hardware_config_path):
            raise RuntimeError("Failed to initialize hardware server.")
        # set to maintain position mode (manipServer.cpp)
        server.set_high_level_maintain_position()

        # image transform
        self.image_transform = get_image_transform(
            input_res=input_res, output_res=camera_res_hw, bgr_to_rgb=False
        )

        # check if bimanual
        if server.is_bimanual():
            id_list = [0, 1]
        else:
            id_list = [0]

        # set force control parameters
        # Tr: transformation matrix from force sensor frame to end-effector frame
        # n_af: number of active force control dimensions
        # default_stiffness: default stiffness matrix for force control
        # compliant_dimensionality: number of compliant dimensions (1 to 6)
        Tr = np.eye(6)
        n_af = compliant_dimensionality
        default_stiffness = [5000, 5000, 5000, 100, 100, 100]
        default_stiffness = np.diag(default_stiffness)

        ft_filter = []
        rgb_row_combined_buffer = []
        rgb_buffer = []
        output_rgb_buffer = []
        for id in id_list:
            server.set_force_controlled_axis(Tr, n_af, id)
            server.set_stiffness_matrix(default_stiffness, id)
            ft_filter.append(
                LiveLPFilter(
                    fs=filter_params["sampling_freq"],
                    cutoff=filter_params["cutoff_freq"],
                    order=filter_params["order"],
                    dim=6,
                )
            )
            # initialize rgb buffers
            # (c h) (n w)->n h w c
            # PIPELINE: Hardware → Raw Buffer → Processing Buffer → Application Buffer → Model
            #                      (packed 2D)       (raw 4D)        (processed 4D)
            rgb_row_combined_buffer.append(
                np.zeros(
                    (input_res[0] * 3, input_res[1] * query_sizes["rgb"]),
                    dtype=np.uint8,
                )
            )
            rgb_buffer.append(
                np.zeros((query_sizes["rgb"], *input_res, 3), dtype=np.uint8)
            )
            output_rgb_buffer.append(
                np.zeros(
                    (query_sizes["rgb"], camera_res_hw[0], camera_res_hw[1], 3),
                    dtype=np.uint8,
                )
            )

        self.server = server
        self.query_sizes = query_sizes
        self.id_list = id_list
        self.ft_filter = ft_filter
        self.rgb_row_combined_buffer = rgb_row_combined_buffer
        self.rgb_buffer = rgb_buffer
        self.output_rgb_buffer = output_rgb_buffer
        self.rgb_timestamp_s = [np.array] * len(id_list)
        self.ts_pose_fb = [np.array] * len(id_list)
        self.ts_pose_fb_timestamp_s = [np.array] * len(id_list)
        self.wrench = [np.array] * len(id_list)
        self.wrench_filtered = [np.array] * len(id_list)
        self.wrench_timestamp_s = [np.array] * len(id_list)

        print("[ManipServerEnv] Waiting for hardware server to be ready...")
        while not self.server.is_ready():
            time.sleep(0.1)

    @property
    def current_hardware_time_s(self):
        return self.server.get_timestamp_now_ms() / 1000

    @property
    def camera_ids(self):
        return self.id_list

    def reset(self):
        # do nothing
        pass

    def cleanup(self):
        pass
        # self.server.join_threads()

    def schedule_controls(
        self,
        pose7_cmd: np.ndarray,  # 7xN or 14xN
        timestamps: np.ndarray,  # 1xN
        stiffness_matrices_6x6: np.ndarray = None,  # 6x(6xN), or 12x(6xN)
    ):
        if len(self.id_list) == 1:
            assert pose7_cmd.shape[0] == 7
        else:
            assert pose7_cmd.shape[0] == 14

        assert timestamps.shape[0] == pose7_cmd.shape[1]

        if not self.server.is_ready():
            return False

        # schedule virtual target to execute in the robot
        for id in self.id_list:
            self.server.schedule_waypoints(
                pose7_cmd[id * 7 : (id + 1) * 7, :], timestamps, id
            )

        # schedule stiffness to execute in the robot
        if stiffness_matrices_6x6 is not None:
            if len(self.id_list) == 1:
                assert stiffness_matrices_6x6.shape[0] == 6
            else:
                assert stiffness_matrices_6x6.shape[0] == 12
            assert stiffness_matrices_6x6.shape[1] == timestamps.shape[0] * 6

            for id in self.id_list:
                self.server.schedule_stiffness(
                    stiffness_matrices_6x6[6 * id : 6 * (id + 1), :], timestamps, id
                )

        return True

    def get_observation_from_buffer(self):
        """Get observations from hardware server buffer.
        The number of data points will be sufficient for later downsampling according
        to the shape meta.

        rgb output: (n, H, W, C)
        """
        # read data from buffer
        for id in self.id_list:
            self.rgb_row_combined_buffer[id][:] = self.server.get_camera_rgb(
                self.query_sizes["rgb"], id
            )
            self.rgb_timestamp_s[id] = (
                self.server.get_camera_rgb_timestamps_ms(id) / 1000
            )

        for id in self.id_list:
            self.ts_pose_fb[id] = self.server.get_pose(
                self.query_sizes["ts_pose_fb"], id
            ).transpose()
            self.ts_pose_fb_timestamp_s[id] = (
                self.server.get_pose_timestamps_ms(id) / 1000
            )

            self.wrench[id] = self.server.get_wrench(
                self.query_sizes["wrench"], id
            ).transpose()
            self.wrench_timestamp_s[id] = (
                self.server.get_wrench_timestamps_ms(id) / 1000
            )

        timedebug0 = time.perf_counter()
        #  process data
        for id in self.id_list:
            # This RGB processing part takes about 60ms
            # 4D to final image
            self.rgb_buffer[id][:] = rearrange(
                self.rgb_row_combined_buffer[id],
                "(c h) (n w)->n h w c",
                c=3,
                n=self.query_sizes["rgb"],
            )
            assert self.rgb_buffer[id].shape == (self.query_sizes["rgb"], *input_res, 3)
            for i in range(self.query_sizes["rgb"]):
                self.output_rgb_buffer[id][i] = self.image_transform(
                    self.rgb_buffer[id][i]
                )

            # wrench = -wrench  # flip the sign for FT300
            self.wrench_filtered[id] = np.zeros_like(self.wrench[id])

            # This filtering part takes about 45ms for 7000 data points
            # filter wrench
            for i in range(self.query_sizes["wrench"]):
                self.wrench_filtered[id][i] = self.ft_filter[id](self.wrench[id][i])

            assert self.ts_pose_fb[id].shape == (self.query_sizes["ts_pose_fb"], 7)
            assert self.wrench[id].shape == (self.query_sizes["wrench"], 6)

        timedebug1 = time.perf_counter()
        print(f"get obs: Time for processing data: {timedebug1 - timedebug0}")

        results = {}
        for id in self.id_list:
            results[f"rgb_{id}"] = self.output_rgb_buffer[id]
            results[f"rgb_time_stamps_{id}"] = self.rgb_timestamp_s[id]
            results[f"ts_pose_fb_{id}"] = self.ts_pose_fb[id]
            results[f"robot_time_stamps_{id}"] = self.ts_pose_fb_timestamp_s[id]
            results[f"wrench_{id}"] = self.wrench_filtered[id]
            results[f"wrench_time_stamps_{id}"] = self.wrench_timestamp_s[id]

            # debug
            print(f"self.wrench_filtered[{id}]: ", self.wrench_filtered[id][-1])

        # check timing
        for id in self.id_list:
            dt_rgb = self.current_hardware_time_s - results[f"rgb_time_stamps_{id}"][-1]
            dt_ts_pose = (
                self.current_hardware_time_s - results[f"robot_time_stamps_{id}"][-1]
            )
            dt_wrench = (
                self.current_hardware_time_s - results[f"wrench_time_stamps_{id}"][-1]
            )
            print(
                f"[get obs] obs lagging for robot {id}: dt_rgb: {dt_rgb}, dt_ts_pose: {dt_ts_pose}, dt_wrench: {dt_wrench}"
            )
        return results
