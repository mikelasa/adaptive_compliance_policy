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
pip install --use-pep517 ur-rtde==1.5.6
pip install v4l2py
pip install toppra
```
4. 
