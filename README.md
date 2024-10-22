# adaptive_compliance_policy


## Install
The following is tested on Ubuntu 22.04.

1. Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
2. Clone this repo along with its submodule.
``` sh
git clone --recursive git@github.com:yifan-hou/adaptive_compliance_policy.git
```
3. Create a virtual env called `pyrite`:
``` sh
cd adaptive_compliance_policy
mamba env create -f conda_environment.yaml
# after finish, activate it using
mamba activate pyrite
# a few pip installs
pip install v4l2py
pip install toppra
pip install atomics
pip install vit-pytorch # Need at least 1.7.12, which was not available in conda
```
4. 

## Setup robot controllers (C++)
### Pull the code
``` sh
# https://github.com/yifan-hou/cpplibrary
git clone git@github.com:yifan-hou/cpplibrary.git
# https://github.com/yifan-hou/force_control
git clone git@github.com:yifan-hou/force_control.git
# https://github.com/yifan-hou/hardware_interfaces
git clone git@github.com:yifan-hou/hardware_interfaces.git
# https://github.com/yifan-hou/RobotTestBench
git clone git@github.com:yifan-hou/RobotTestBench.git
```

### Build & Install
In each of the package above, run the following:
``` sh
cd package
mkdir build && cd build
cmake ..
make -j
make install # except for RobotTestBench
```

### (Optional) Install to a local path
I recommend to install to a local path, so you don't need sudo access. To do so, build & install the packages but replace the line
``` sh
cmake ..
```
with
``` sh
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local  ..
```
where `$HOME/.local` can be replaced with any local path.

Then you need to tell gcc to look for binaries/headers from your local path by adding the following to your .bashrc or .zshrc:
``` sh
export PATH=$HOME/.local/bin:$PATH
export C_INCLUDE_PATH=$HOME/.local/include/:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$HOME/.local/include/:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/:$LD_LIBRARY_PATH
```
You need to run `source .bashrc` or reopen a terminal for those to take effect.

