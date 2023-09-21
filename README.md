# This repository contains various Maximum Entropy Reinforcement Learning algorithms.
* Link to my notes on the topic (maths): [MaxEntRL](https://drive.google.com/file/d/1H9TMM-vtvaI10YVXEul7F1-CC6D1AXqD/view?usp=sharing).

## Disclaimers:
* Although this is not a fork or a clone of any other implementation of any other algorithms, the code design has been inspired by the [rlkit](https://github.com/rail-berkeley/rlkit) repo. Thanks to the contributors of rlkit for providing a [PyTorch](https://github.com/pytorch/pytorch) implementation of [SAC](https://arxiv.org/abs/1812.05905).

## Implemented Algorithms:
* You can check the settable parameters for the experiments in the  `params_func_config` dictionary in the [experiments_init_utils.py](https://github.com/mariovas3/MaxEntRL/rl_algos/algorithms/experiments_init_utils.py) file.
* Soft Actor-Critic, tested on the "Hopper-v2" environment from [MuJoCo](https://mujoco.org/).
    * Example test script is to run the following command from 
    root of repo:
        * `python tests/test_hopperv2.py env_name=Hopper-v2 num_iters=100 net_hiddens=256,256 num_policy_updates=1 num_qfunc_updates=5 max_size=100000 policy_lr=1e-3 qfunc_lr=1e-3 temperature_lr=1e-3 batch_size=250 policy_epochs=10`

## MuJoCo env setup tested on Ubuntu 20.04:
* `pip install -U portalocker`
* `pip install -U lockfile`
* `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<your_name>/.mujoco/mjpro150/bin  # put this in the ~/.bashrc and source the file;`
* `sudo apt-get install libosmesa6-dev  # fix the missing GL/osmesa.h file error;`
* `sudo apt-get install patchelf  # fix no such file patchelf error;`
* Provided you have downloaded mjpro150 and have an access key the following should install `mujoco-py`:
    * `pip install -U 'mujoco-py<1.50.2,>=1.50.1'`
* These were the only extra quirks I needed to fix before I got MuJoCo running locally.
* The above should run smoothly with:
    * `mujoco==2.3.6`
    * `mujoco-py==1.50.1.68`
    * `gymnasium==0.28.1`