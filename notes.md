# High level usages
## Collect training data
Start the demonstration collection script. Press "C" to start recording. Use SpaceMouse to move the robot. Press "S" to stop recording. 
```console
(robodiff)[diffusion_policy]$ python demo_real_robot.py -o data/demo_pusht_real --robot_ip 192.168.0.204
```

This should result in a demonstration dataset in `data/demo_pusht_real` with in the same structure as our example [real Push-T training dataset](https://diffusion-policy.cs.columbia.edu/data/training/pusht_real.zip).


Uses RealEnv to actually save data diffusion_policy/real_world/real_env.py
    env.start_episode
        start three data accumulators
    env.end_episode
        read the data in the accumulators in the replybuffer, replybuffer save them to disk
            self.replay_buffer.add_episode(episode, compressors='disk')
        img data is saved in a video next to the zarr file.
            in_zarr_path = input.joinpath('replay_buffer.zarr')
            in_video_dir = input.joinpath('videos')

## Train
```sh
python3 train.py --config-name=train_diffusion_transformer_hybrid_workspace task.dataset_path=data/demo_pusht_real
# A more practical way:
clear;  HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name=train_diffusion_unet_timm_flip_up_workspace;
```

uses real_data_conversion.py to load data into a replay_buffer, it also load the video and add it to the replay_buffer.
(pusht_image_dataset)
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255

**shape_meta**
* task yaml contains "shape_meta", which specifies the low_dim keys and image keys to use. It is used in dataset.py and needs to be consistent with the actual data format.
* shape_meta is organized by 'obs' and 'action', where 'obs' could have both low_dim fields and image fields. Both 'action' and the low_dim 'obs' fields should be extracted from zarr. The extraction of all low_dim keys from meta_data is done in dataset.py. 

real_env and replay_buffer themselves do not have assumption on what keys are contained.

## Evaluation
Assuming the training has finished and you have a checkpoint at `data/outputs/blah/checkpoints/latest.ckpt`, launch the evaluation script with:
```console
python eval_real_robot.py -i data/outputs/blah/checkpoints/latest.ckpt -o data/eval_pusht_real --robot_ip 192.168.0.204
```
Press "C" to start evaluation (handing control over to the policy). Press "S" to stop the current episode.


Uses RealEnv diffusion_policy/real_world/real_env.py


# data format
1. zarr data
Exists: on file system

Keys:
```
    data/
        episodes/
            raw_keys
            low_dim_time_stamps
            visual_time_stamps

    meta/
        episode_low_dim_len
        episode_rgb_len
```
Used by: replay_buffer.py

2. replay_buffer_raw (raw)
Exists: in task_dataset.py as a temporary variable
Obtained by loading data with replay_buffer class

Keys follow zarr data above

3. replay_buffer (obs)
convert to obs keys, create 'action', 'action_time_stamps'
action_time_stamps is the same as low_dim_time_stamps
Exists: in task_dataset.py as a member variable.
Obtained by self.raw_episodes_conversion()
Require task specific conversion in task_dataset.py and task_type_conversion.py

size: no subsampling or padding, full length data
action and low_dim has the same size.

keys:
```
    data/
        episodes/
            obs/
                obs_keys
                low_dim_time_stamps
                visual_time_stamps
            action
            action_time_stamps
    meta/
        episode_low_dim_len
        episode_rgb_len
```


4. sample
Exists: when sampling_sequence is called. Created by sampler.py
involves sample specific computation (e.g. convert to local frame) to a window of data.

Used in training. Implemented in sampler, dataset/get_normalizer.
Each entry has a separate normalizer, even when there are duplicates

``` yaml
# rgb vs. low_dim: raw data are either rgb images or low_dim vectors. frames are aligned within each type.
# obs vs. action: obs data are used as policy input; action are used as labels for policy output.

# down_sample_steps: how many steps to skip in the raw data for the given usage
# horizon: how many steps to look ahead(action) or back(obs) after downsample for the given usage
sparse_obs_rgb_down_sample_steps: 10
sparse_obs_rgb_horizon: 1

sparse_obs_low_dim_down_sample_steps: 1
sparse_obs_low_dim_horizon: 2

sparse_action_down_sample_steps: 10
sparse_action_horizon: 16
```

```
sample:
    obs:
      sparse:
        camera1_rgb:
          horizon: ${task.img_obs_horizon} # int
          down_sample_steps: ${task.sparse_obs_down_sample_steps} # int
        robot0_eef_pos:
          horizon: ${task.low_dim_obs_horizon} # int
          down_sample_steps: ${task.sparse_obs_down_sample_steps} # float
    action:
      sparse:
        horizon: ${task.sparse_action_horizon}
        down_sample_steps: ${task.sparse_action_down_sample_steps} # int
```

# normalizer
``` py
normalizer['sparse_action']
```

# data flow for training
### `data`
saved on disk. It is one folder saved from zarr.

### `Dataset class`
__init__: load the zarr dataset, save into a replaybuffer (optionally create a cache)
sampler: read one entry from dataset. Uses horizon to decide how many timesteps to read
get_item: read one entry, convert it to a format consistent to the `shape_meta`
get_normalizer: read all data, exam each key and compute normalizer for it
    * requirements on shape_meta: normalizer works on a dict. Each value of the dict must be array, not another dict. All keys must be different.

### Workspace
Instantiate Dataset. It uses hydra, so the arguments to __init__ come from config (task.dataset)




On the task side, we have:
* `EnvRunner`: wrapper to executes a `Policy` in an Env and produce logs and metrics (i.e. runs simulation)
* `config/task/<task_name>.yaml`: contains all information needed to construct `Dataset` and `EnvRunner`.
    Notably, the shape_meta.
* (optional) `Env`: an `gym==0.21.0` compatible class that encapsulates the task environment.

On the policy side, we have:
* `Policy`: implements inference according to the interface and part of the training process.
    obs_encoder: all observations go through obs_encoder, gets aggregated into obs
* `Workspace`: manages the life-cycle of training and evaluation (interleaved) of a method. 
* `config/<workspace_name>.yaml`: contains all information needed to construct `Policy` and `Workspace`.


**Training**
python train.py --config-name=train_diffusion_unet_real_image_workspace task.dataset_path=data/demo_pusht_real
* train.py create a *workspace* from config
* In workspace
    * create a *dataset* from config
    * create a *EnvRunner* from config (optional, just for create some logs)
    * create a *Policy* from config

**Evaluation**
python eval_real_robot.py -i data/outputs/blah/checkpoints/latest.ckpt -o data/eval_pusht_real --robot_ip 192.168.0.204
* eval_real_robot.py creates a workspace from the checkpoint
* eval_real_robot.py creates a policy from the checkpoint
* eval_real_robot.py creates a RealEnv

# Hacky notes

## shape_meta is different from data shape
In umi.yaml, some shape_meta entries has a "raw_shape" entry in addition to "shape". This 
is because the Dataset class will modify the data entry when loading them. 
* shape: the shape of data presented by Dataset.
* raw_shape: the shape of data in the zarr storage.

In fact, shape_meta is mostly used to describe the data after loaded by the dataset classes.
It might not corresponds properly to the zarr storage. For example, the shape meta
"robot0_eef_rot_axis_angle_wrt_start" is completely created in the dataset class, based on
some entries in zarr that are not listed in shape meta.

## Computing poses "wrt start pose"
This is to give the policy a concept of where the base is, so as to properly tell concepts
like "left or right" in the camera view.

A side effect is that there are duplicate information in the shape_meta (two eef_pose),
which is fine for nn policies.


## latency matching in data vs. inference
You can add latency in data (see shape meta), so the training will consider latency matching.
The other option is to add latency matching in inference. This has the advantage of flexibility
for different latencies. Downside is to drop data frames for the faster modality.

Default uses inference latency matching only. The latency in shape_meta is set to zero.


## What to update when action is changed
* xxxtask_dataset.py
* xxxtask.yaml (shape_meta)
* train_xxx_workspace.py/log_action_mse



# Question
