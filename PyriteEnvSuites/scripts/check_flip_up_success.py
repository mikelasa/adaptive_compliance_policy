import zarr
import numpy as np
import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

from PyriteUtility.data_pipeline.episode_data_buffer import (
    VideoData,
    EpisodeDataBuffer,
    EpisodeDataIncreImageBuffer,
)
from spatialmath.base import q2r
import concurrent.futures
from utils.flip_up_env_utils import is_flip_up_success

dataset_path = "/path/to/your/dataset"
buffer = zarr.open(dataset_path)

success_id = []
failure_id = []


angle_thresh = 10
for i in range(len(buffer["data"])):
    episode = f"episode_{i}"
    ep_data = buffer["data"][episode]
    obj_poses = ep_data["low_dim_state"]

    N, D = obj_poses.shape
    assert D == 7

    success, angle_deg = is_flip_up_success(obj_poses[-1], angle_thresh)

    if success:
        success_id.append(i)
    else:
        failure_id.append(i)

    print(f"{episode}, \tangle: {angle_deg:.2f}")

print(f"Success: {len(success_id)}, Failure: {len(failure_id)}")
print(f"Success rate: {len(success_id) / (len(success_id) + len(failure_id)):.2f}")

# copy successful episodes to a new dataset
dataset_path_new = dataset_path + "_filtered"

store = zarr.DirectoryStore(path=dataset_path_new)
root = zarr.open(store=store, mode="w")
data_output = root.create_group("data")


def copy_one_episode(input_data_group, output_data_group, i):
    episode = f"episode_{i}"
    ep_data = input_data_group[episode]
    ep_data_output = output_data_group.create_group(episode)
    print(f"Copying {episode}...")

    for key in ep_data.keys():
        value = ep_data[key]
        if isinstance(value, zarr.Array):
            n_copied, n_skipped, n_bytes_copied = zarr.copy(
                source=value,
                dest=ep_data_output,
                name=key,
                chunks=value.chunks,
                compressor=value.compressor,
                if_exists="replace",
            )
        else:
            ep_data_output.array(
                name=key, data=value, chunks=value.chunks, compressor=value.compressor
            )


with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    futures = set()
    for i in success_id:
        futures.add(executor.submit(copy_one_episode, buffer["data"], data_output, i))

    concurrent.futures.wait(futures)

print("Done!")
