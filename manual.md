
# Build
``` sh
conda activate umi
export LD_LIBRARY_PATH=/path/to/anaconda_or_miniforge/envs/pyrite/lib/:$LD_LIBRARY_PATH
```

# Inference
1. [RobotTestBench] change hardware config to inference style
2. [PyriteEnvSuites] In env runner, specify the checkpoint
3. Run the env runner in umi env.