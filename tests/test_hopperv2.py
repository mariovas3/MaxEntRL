import sys
from pathlib import Path
p = Path(__file__).absolute().parent.parent
if sys.path[-1] != str(p):
    sys.path.append(str(p))
print(str(p))

from rl_algos.algorithms.sac import SACAgent
from rl_algos.algorithms.experiments_init_utils import *
from rl_algos.algorithms.vis_utils import see_one_episode

from functools import partial

import random
import numpy as np
import torch


if __name__ == "__main__":
    # set training params from stdin;
    # NOTE: booleans are passed as 0 for False and other int for True;
    params_func_config = arg_parser(params_func_config, sys.argv)
    
    # set the seed;
    random.seed(params_func_config['seed'])
    np.random.seed(params_func_config['seed'])
    torch.manual_seed(params_func_config['seed'])

    
    # RL train config;
    get_params_train = partial(get_params, **params_func_config)

    # get kwargs;
    agent_kwargs, config = get_params_train()
    
    # init agent for the IRL training;
    agent = SACAgent(
        **agent_kwargs,
        **config
    )
    
    # start RL training;
    print(f"RL training for {params_func_config['policy_epochs']} epochs")
    
    # train RL;
    agent.train_k_epochs(k=params_func_config['policy_epochs'], 
                         config=params_func_config)

    env = gym.make(params_func_config['env_name'], render_mode='human')
    see_one_episode(env, agent, seed=0, save_to=agent.save_to)
