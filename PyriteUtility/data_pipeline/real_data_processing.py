import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

from PyriteUtility.data_pipeline.episode_data_buffer import (
    VideoData,
    EpisodeDataBuffer,
)
from PyriteUtility.data_pipeline.timestamp_repair import (
    PHASE_CONCENTRATION_MIN,
    repair_timestamps_from_seq_id,
)
import scipy.signal

import pathlib
import shutil
import numpy as np
import pandas as pd
import zarr
import cv2
import concurrent.futures

# check environment variables
if "PYRITE_RAW_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_RAW_DATASET_FOLDERS")
if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")


def _repaired_time_stamps(df, time_stamp_column, episode_name, stream_name):
    """Re-derive a hardware-clocked stream's sample times from its seq_id.

    The recorded times are host receive times, so a stall in the logging thread
    leaves a hole in the stream followed by a burst of samples sharing one
    timestamp -- see PyriteUtility/data_pipeline/timestamp_repair.py.

    Only streams that are actually hardware-clocked get repaired. The F/T stream
    is; the Franka robot stream is not (its timestamps are arrival times whose
    scatter is real), so it passes through untouched. Raw data captured before
    seq_id was logged also passes through unchanged.
    """
    raw = df[time_stamp_column].to_numpy().astype(np.float64)
    if "seq_id" not in df.columns:
        print(
            f"  {episode_name} {stream_name}: no seq_id in raw data, "
            f"leaving {time_stamp_column} as recorded"
        )
        return raw

    repaired, info = repair_timestamps_from_seq_id(df["seq_id"].to_numpy(), raw)
    if info["phase_concentration"] < PHASE_CONCENTRATION_MIN:
        # Arrival-stamped, not hardware-clocked: there is no finer sample time
        # hiding behind these values to recover.
        return raw
    if info["n_stall_intervals"]:
        print(
            f"  {episode_name} {stream_name}: repaired {info['n_stall_intervals']} "
            f"stall interval(s), max correction {info['max_correction']:.2f} ms "
            f"({info['method']}, {info['n'] - info['n_clean']} of {info['n']} "
            f"samples contaminated)"
        )
    if not info["seq_contiguous"]:
        print(
            f"  {episode_name} {stream_name}: WARNING seq_id is not contiguous -- "
            f"the hardware genuinely dropped samples here"
        )
    return repaired


def image_read(rgb_dir, rgb_file_list, i, output_data_rgb, output_data_rgb_time_stamps):
    img_name = rgb_file_list[i]
    img = cv2.imread(str(rgb_dir.joinpath(img_name)))
    # convert BGR to RGB for imageio
    output_data_rgb[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    time_img_ms = float(img_name[11:22])
    output_data_rgb_time_stamps[i] = time_img_ms
    return True


# specify the input and output directories
TWO_CAMERAS = False  # set True to include second camera (camera_id=1)

id_list = [0]  # single robot
# id_list = [0, 1] # bimanual
camera_id_list = [0, 1] if TWO_CAMERAS else [0]

input_dir = pathlib.Path(
    os.environ.get("PYRITE_RAW_DATASET_FOLDERS") + "/CAP5/curva"
)
output_dir = pathlib.Path(os.environ.get("PYRITE_DATASET_FOLDERS") + "/CAP5/todo")

robot_timestamp_dir = output_dir.joinpath("robot_timestamp")
wrench_timestamp_dir = output_dir.joinpath("wrench_timestamp")
rgb_timestamp_dir = output_dir.joinpath("rgb_timestamp")

# clean and create output folders
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)


# # check for black images
# def check_black_images(rgb_file_list, rgb_dir, i, prefix):
#     f = rgb_file_list[i]
#     img = cv2.imread(str(rgb_dir.joinpath(f)))
#     # print the mean of the image
#     img_mean = np.mean(img)
#     if img_mean < 50:
#         print(f"{prefix}, {f} has mean value of {img_mean}")
#     return True


# for episode_name in os.listdir(input_dir):
#     if episode_name.startswith("."):
#         continue

#     episode_dir = input_dir.joinpath(episode_name)
#     for id in id_list:
#         rgb_dir = episode_dir.joinpath("rgb_" + str(id))
#         rgb_file_list = os.listdir(rgb_dir)
#         num_raw_images = len(rgb_file_list)
#         print(f"Checking for black images in {rgb_dir}")
#         with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
#             futures = set()
#             for i in range(len(rgb_file_list)):
#                 futures.add(
#                     executor.submit(
#                         check_black_images,
#                         rgb_file_list,
#                         rgb_dir,
#                         i,
#                         f"{episode_name} rgb_{id}",
#                     )
#                 )

#             completed, futures = concurrent.futures.wait(futures)
#             for f in completed:
#                 if not f.result():
#                     raise RuntimeError("Failed to read image!")
# exit()


# open the zarr store
store = zarr.DirectoryStore(path=output_dir)
root = zarr.open(store=store, mode="a")

print("Reading data from input_dir: ", input_dir)
episode_names = os.listdir(input_dir)


def process_one_episode(root, episode_name, input_dir, id_list, camera_id_list):
    # ignore hidden files and stray artifacts that are not actual episodes
    if episode_name.startswith("."):
        return True

    # info about input
    episode_id = episode_name[8:]
    print(f"episode_name: {episode_name}, episode_id: {episode_id}")
    episode_dir = input_dir.joinpath(episode_name)

    # skip anything that is not a directory or does not match expected naming
    if not episode_dir.is_dir():
        print(f"Skipping non-directory entry: {episode_dir}")
        return True
    if not episode_name.startswith("episode_"):
        print(f"Skipping non-episode entry: {episode_dir}")
        return True

    # read rgb
    data_rgb = []
    data_rgb_time_stamps = []
    rgb_data_shapes = []
    for id in camera_id_list:
        rgb_dir = episode_dir.joinpath("rgb_" + str(id))
        rgb_file_list = os.listdir(rgb_dir)
        rgb_file_list.sort()  # important!
        num_raw_images = len(rgb_file_list)
        img = cv2.imread(str(rgb_dir.joinpath(rgb_file_list[0])))

        rgb_data_shapes.append((num_raw_images, *img.shape))
        data_rgb.append(np.zeros(rgb_data_shapes[id], dtype=np.uint8))
        data_rgb_time_stamps.append(np.zeros(num_raw_images))

        print(f"Reading rgb data from: {rgb_dir}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = set()
            for i in range(len(rgb_file_list)):
                futures.add(
                    executor.submit(
                        image_read,
                        rgb_dir,
                        rgb_file_list,
                        i,
                        data_rgb[id],
                        data_rgb_time_stamps[id],
                    )
                )

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to read image!")

    # read low dim data
    data_ts_pose_fb = []
    data_robot_time_stamps = []
    data_wrench = []
    data_wrench_time_stamps = []
    print(f"Reading low dim data for : {episode_dir}")
    for id in id_list:
        json_path = episode_dir.joinpath("robot_data_" + str(id) + ".json")
        df_robot_data = pd.read_json(json_path)
        data_robot_time_stamps.append(
            _repaired_time_stamps(
                df_robot_data, "robot_time_stamps", episode_name, f"robot_{id}"
            )
        )
        data_ts_pose_fb.append(np.vstack(df_robot_data["ts_pose_fb"]))

        # read wrench data
        json_path = episode_dir.joinpath("wrench_data_" + str(id) + ".json")
        df_wrench_data = pd.read_json(json_path)
        data_wrench_time_stamps.append(
            _repaired_time_stamps(
                df_wrench_data, "wrench_time_stamps", episode_name, f"wrench_{id}"
            )
        )
        data_wrench.append(np.vstack(df_wrench_data["wrench"]))

    # get filtered force
    print(f"Computing filtered wrench for {episode_name}")
    # This used to be LiveLPFilter(fs=500, cutoff=5, order=5) fed one sample at
    # a time. Two things were wrong with that here:
    #
    # 1. fs was hardcoded to 500, which was right for the 500 Hz UR5 of the
    #    original work but silently doubled the corner to 10 Hz on a 1 kHz
    #    Franka -- so the stiffness labels have been carrying twice the force
    #    noise they were designed to. The rate is now read off the timestamps,
    #    per arm, so it stays correct on either robot.
    # 2. LiveLPFilter is causal because it exists for the online control path.
    #    Here the wrench is an already-recorded array, so causality buys
    #    nothing and the phase lag is pure loss: it delayed the k label 51-88 ms
    #    behind the force that produced it. filtfilt runs the filter forwards
    #    and backwards, giving the same attenuation with zero phase.
    #
    # Note this does not remove oscillations below the corner. A ~4 Hz contact
    # instability still reaches the label almost intact; see
    # PyriteEnvSuites/scripts/analyze_label_filter_cutoff.py to measure what a
    # given episode actually passes through.
    FORCE_FILTER_CUTOFF_HZ = 5
    FORCE_FILTER_ORDER = 5
    data_wrench_filtered = []
    for id in id_list:
        # timestamps are in milliseconds throughout this pipeline
        dt_ms = np.median(np.diff(data_wrench_time_stamps[id]))
        fs = 1000.0 / dt_ms
        sos = scipy.signal.butter(
            FORCE_FILTER_ORDER, FORCE_FILTER_CUTOFF_HZ, fs=fs, output="sos"
        )
        data_wrench_filtered.append(
            scipy.signal.sosfiltfilt(sos, data_wrench[id], axis=0)
        )
        print(f"  arm {id}: wrench at {fs:.0f} Hz, "
              f"zero-phase lowpass at {FORCE_FILTER_CUTOFF_HZ} Hz")

    # make time stamps start from zero
    time_offsets = []
    for id in camera_id_list:
        time_offsets.append(data_rgb_time_stamps[id][0])
    for id in id_list:
        time_offsets.append(data_robot_time_stamps[id][0])
        time_offsets.append(data_wrench_time_stamps[id][0])
    time_offset = np.min(time_offsets)
    for id in camera_id_list:
        data_rgb_time_stamps[id] -= time_offset
    for id in id_list:
        data_robot_time_stamps[id] -= time_offset
        data_wrench_time_stamps[id] -= time_offset

    # create output zarr
    print(f"Saving everything to : {output_dir}")
    recoder_buffer = EpisodeDataBuffer(
        store_path=output_dir,
        camera_ids=camera_id_list,
        save_video=True,
        save_video_fps=60,
        data=root,
    )

    # save data using recoder_buffer
    rgb_data_buffer = {}
    for id in camera_id_list:
        rgb_data = data_rgb[id]
        rgb_data_buffer.update({id: VideoData(rgb=rgb_data, camera_id=id)})
    recoder_buffer.create_zarr_groups_for_episode(rgb_data_shapes, camera_id_list, episode_id)
    recoder_buffer.save_video_for_episode(
        visual_observations=rgb_data_buffer,
        visual_time_stamps=data_rgb_time_stamps,
        episode_id=episode_id,
    )
    recoder_buffer.save_low_dim_for_episode(
        ts_pose_command=data_ts_pose_fb,
        ts_pose_fb=data_ts_pose_fb,
        wrench=data_wrench,
        wrench_filtered=data_wrench_filtered,
        robot_time_stamps=data_robot_time_stamps,
        wrench_time_stamps=data_wrench_time_stamps,
        episode_id=episode_id,
    )
    return True

"""
with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(
            process_one_episode,
            root,
            episode_name,
            input_dir,
            id_list,
            camera_id_list,
        )
        for episode_name in episode_names
    ]
    for future in concurrent.futures.as_completed(futures):
        if not future.result():
            raise RuntimeError("Multi-processing failed!")

"""
for episode_name in episode_names:
    process_one_episode(root, episode_name, input_dir, id_list, camera_id_list)
            
print("Finished reading. Now start generating metadata")
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs

register_codecs()
buffer = zarr.open(output_dir)
meta = buffer.create_group("meta", overwrite=True)
episode_robot_len = []
episode_wrench_len = []
episode_rgb_len = []

for id in id_list:
    episode_robot_len.append([])
    episode_wrench_len.append([])
for id in camera_id_list:
    episode_rgb_len.append([])

count = 0
for key in buffer["data"].keys():
    episode = key
    ep_data = buffer["data"][episode]

    for id in id_list:
        episode_robot_len[id].append(ep_data[f"ts_pose_fb_{id}"].shape[0])
        episode_wrench_len[id].append(ep_data[f"wrench_{id}"].shape[0])
        print(
            f"Number {count}: {episode}: robot id={id}: robot len: {episode_robot_len[id][-1]}, wrench_len: {episode_wrench_len[id][-1]}"
        )
    for id in camera_id_list:
        episode_rgb_len[id].append(ep_data[f"rgb_{id}"].shape[0])
        print(
            f"Number {count}: {episode}: camera id={id}: rgb len: {episode_rgb_len[id][-1]}"
        )
    count += 1

for id in id_list:
    meta[f"episode_robot{id}_len"] = zarr.array(episode_robot_len[id])
    meta[f"episode_wrench{id}_len"] = zarr.array(episode_wrench_len[id])
for id in camera_id_list:
    meta[f"episode_rgb{id}_len"] = zarr.array(episode_rgb_len[id])

print(f"All done! Generated {count} episodes in {output_dir}")
print("The only thing left is to run postprocess_add_virtual_target_label.py")
