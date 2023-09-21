import math
import numpy as np

import torch
from torch import nn

from rl_algos.algorithms.vis_utils import *
from rl_algos.algorithms.rl_utils import OI_init

from typing import Optional
from copy import deepcopy
from pathlib import Path
import time
import pickle
from tqdm import tqdm


# path to save logs;
TEST_OUTPUTS_PATH = Path(__file__).absolute().parent.parent.parent / "test_output"


def track_params(Q1t, Q1, tau):
    """
    Updates parameters of Q1t to be:
    Q1t = Q1 * tau + Q1t * (1 - tau).
    """
    theta = nn.utils.parameters_to_vector(Q1.parameters()) * tau + (
        1 - tau
    ) * nn.utils.parameters_to_vector(Q1t.parameters())

    # load theta as the new params of Q1;
    nn.utils.vector_to_parameters(theta, Q1t.parameters())


class SACAgentBase:
    def __init__(
        self,
        name,
        policy_constructor,
        qfunc_constructor,
        env,
        buffer_constructor,
        optimiser_constructors,
        entropy_lb,
        policy_lr,
        temperature_lr,
        qfunc_lr,
        tau,
        discount,
        save_to: Optional[Path] = TEST_OUTPUTS_PATH,
        cache_best_policy=False,
        clip_grads=False,
        fixed_temperature=None,
        num_policy_updates=None,
        num_qfunc_updates=None,
        **kwargs,
    ):
        self.device = None
        self.name = name
        self.save_to = save_to
        self.clip_grads = clip_grads
        self.num_policy_updates = num_policy_updates
        self.num_qfunc_updates = num_qfunc_updates
        self.fixed_temperature = fixed_temperature
        self.sac_epochs_done = 0

        # experimental:
        # chache best nets;
        self.cache_best_policy = cache_best_policy
        self.best_eval_return = None
        self.best_policy = None
        self.best_Q1, self.best_Q2 = None, None
        self.best_Q1t, self.best_Q2t = None, None
        self.best_log_temp = None

        # training params;
        self.seed = kwargs["training_kwargs"]["seed"]
        self.num_iters = kwargs["training_kwargs"]["num_iters"]
        self.num_steps_to_sample = kwargs["training_kwargs"][
            "num_steps_to_sample"
        ]
        self.num_grad_steps = kwargs["training_kwargs"]["num_grad_steps"]
        self.batch_size = kwargs["training_kwargs"]["batch_size"]
        self.num_eval_steps_to_sample = kwargs["training_kwargs"][
            "num_eval_steps_to_sample"
        ]
        self.min_steps_to_presample = kwargs["training_kwargs"][
            "min_steps_to_presample"
        ]

        # instantiate necessary items;
        self.Q1 = qfunc_constructor(**kwargs["Q1_kwargs"])
        self.Q2 = qfunc_constructor(**kwargs["Q2_kwargs"])
        self.Q1t = qfunc_constructor(**kwargs["Q1t_kwargs"])
        self.Q2t = qfunc_constructor(**kwargs["Q2t_kwargs"])
        self.policy = policy_constructor(**kwargs["policy_kwargs"])

        # don't track grads for target q funcs
        # and set them in eval mode;
        self.Q1t.requires_grad_(False)
        self.Q2t.requires_grad_(False)
        self.Q1t.eval()
        self.Q2t.eval()

        # instantiate buffer and env and check some of their attributes;
        self.buffer = buffer_constructor(**kwargs["buffer_kwargs"])
        self.env = env
        if self.num_eval_steps_to_sample is None:
            self.num_eval_steps_to_sample = self.env.spec.max_episode_steps

        # init temperature and other parameters;
        if self.fixed_temperature:
            self.log_temperature = torch.tensor(
                np.log(self.fixed_temperature)
            )
        else:
            self.log_temperature = torch.tensor(0.0, requires_grad=True)
            self.temperature_optim = optimiser_constructors[
                "temperature_optim"
            ]([self.log_temperature], lr=temperature_lr)
        self.entropy_lb = (
            entropy_lb
            if entropy_lb
            else -kwargs["policy_kwargs"]["action_dim"]
        )
        self.tau = tau
        self.discount = discount

        # instantiate the optimisers;
        self.policy_optim = optimiser_constructors["policy_optim"](
            self.policy.parameters(), lr=policy_lr
        )

        self.Q1_optim = optimiser_constructors["Q1_optim"](
            self.Q1.parameters(), lr=qfunc_lr
        )
        self.Q2_optim = optimiser_constructors["Q2_optim"](
            self.Q2.parameters(), lr=qfunc_lr
        )

        # loss variables;
        self.temperature_loss = None
        self.policy_loss = None
        self.Q1_loss, self.Q2_loss = None, None

        # bookkeeping metrics;
        self.policy_losses = []
        self.temperature_losses = []
        self.Q1_losses, self.Q2_losses = [], []
        if self.fixed_temperature is None:
            self.temperatures = [math.exp(self.log_temperature.item())]
        self.eval_path_returns, self.eval_path_lens = [], []
        self.eval_path_avg_rewards = []
        self.eval_path_avg_log_probs = []
        self.train_path_returns = self.buffer.undiscounted_returns 
        self.train_path_lens = self.buffer.path_lens
        self.train_path_avg_rewards = self.buffer.avg_reward_per_episode

    def OI_init_nets(self):
        OI_init(self.Q1)
        OI_init(self.Q2)
        OI_init(self.Q1t)
        OI_init(self.Q2t)
        OI_init(self.policy)

    def sample_action(self, obs=None, give_density=False):
        pass

    def clear_buffer(self):
        self.buffer.clear_buffer()
        # empty agent bookkeeping;
        self.policy_losses = []
        self.temperature_losses = []
        self.Q1_losses, self.Q2_losses = [], []
        self.temperatures = []
        self.eval_path_returns, self.eval_path_lens = [], []

    def _cache(self):
        self.best_policy = deepcopy(self.policy)
        self.best_Q1 = deepcopy(self.Q1)
        self.best_Q2 = deepcopy(self.Q2)
        self.best_Q1t = deepcopy(self.Q1t)
        self.best_Q2t = deepcopy(self.Q2t)
        self.best_log_temp = deepcopy(self.log_temperature)

    def sample_deterministic(self, obs=None, give_density=False):
        pass

    def get_policy_loss_and_temperature_loss(self, *args, **kwargs):
        pass

    def get_q_losses(self, *args, **kwargs):
        pass

    def track_qfunc_params(self):
        track_params(self.Q1t, self.Q1, self.tau)
        track_params(self.Q2t, self.Q2, self.tau)
    
    def update_policy_and_temp(self):
        self.policy_optim.zero_grad()
        self.policy_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )
        self.policy_optim.step()

        # update temperature;
        if self.fixed_temperature is None:
            self.temperature_optim.zero_grad()
            self.temperature_loss.backward()
            self.temperature_optim.step()

            # add new temperature to list;
            self.temperatures.append(math.exp(self.log_temperature.item()))

    def update_qfuncs(self):
        # update qfunc1
        self.Q1_optim.zero_grad()
        self.Q1_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(
                self.Q1.parameters(), 
                max_norm=1.0, 
                error_if_nonfinite=True
            )
        self.Q1_optim.step()

        # update qfunc2
        self.Q2_optim.zero_grad()
        self.Q2_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(
                self.Q2.parameters(), 
                max_norm=1.0, 
                error_if_nonfinite=True
            )
        self.Q2_optim.step()

    def check_presample(self):
        if self.min_steps_to_presample:
            self.policy.eval()
            self.buffer.collect_path(
                self.env, self, self.min_steps_to_presample
            )
            self.min_steps_to_presample = 0

    def train_one_epoch(self, *args, **kwargs):
        pass

    def train_k_epochs(
        self,
        k,
        *args,
        config=None,
        **kwargs,
    ):
        for _ in tqdm(range(k)):
            self.sac_epochs_done += 1
            self.train_one_epoch(*args, **kwargs)
        
        # check if have to save;
        if self.save_to is not None:
            metric_names = [
                "policy-loss",
                "qfunc1-loss",
                "qfunc2-loss",
                "train-path-avg-rewards",
                "train-path-avg-rewards-ma-30",
                "train-path-returns",
                "train-returns-ma-30",
                "train-path-lens",
                "train-path-lens-ma-30",
                "eval-path-avg-log-prob",
                "eval-path-avg-log-prob-ma-30",
                "eval-path-avg-rewards",
                "eval-path-avg-rewards-ma-30",
                "eval-path-returns",
                "eval-returns-ma-30",
                "eval-path-lens",
                "eval-path-lens-ma-30",
            ]
            metrics = [
                self.policy_losses,
                self.Q1_losses,
                self.Q2_losses,
                self.train_path_avg_rewards,
                get_moving_avgs(self.train_path_avg_rewards, 30),
                self.train_path_returns,
                get_moving_avgs(self.train_path_returns, 30),
                self.train_path_lens,
                get_moving_avgs(self.train_path_lens, 30),
                self.eval_path_avg_log_probs,
                get_moving_avgs(self.eval_path_avg_log_probs, 30),
                self.eval_path_avg_rewards,
                get_moving_avgs(self.eval_path_avg_rewards, 30),
                self.eval_path_returns,
                get_moving_avgs(self.eval_path_returns, 30),
                self.eval_path_lens,
                get_moving_avgs(self.eval_path_lens, 30),
            ]
            if self.fixed_temperature is None:
                metric_names.extend(["temperature-loss", "temperatures"])
                metrics.append(self.temperature_losses)
                metrics.append(self.temperatures)

            if self.best_policy is not None:
                self.policy = self.best_policy
                self.Q1 = self.best_Q1
                self.Q2 = self.best_Q2
                self.Q1t = self.best_Q1t
                self.Q2t = self.best_Q2t
                self.log_temperature = self.best_log_temp
                print(f"final eval with best policy")

            self.policy.eval()
            (
                r,
                log_probs,
                code,
                ep_len,
            ) = self.buffer.get_single_ep_rewards_and_logprobs(
                self.env, self,
            )
            assert ep_len == len(r)
            if isinstance(r, list):
                r = sum(r)
            else:
                r = r.sum().item()
            print(f"code from sampling eval episode: {code}\n"
                  f"episode len: {ep_len}")

            save_metrics(
                self.save_to,
                metric_names,
                metrics,
                self.name,
                self.env.spec.id,
                self.seed,
                config=config,
                last_eval_rewards=None,
                suptitle=(
                    f"figure after {self.sac_epochs_done}"
                    " SAC Epochs"
                ),
            )


def save_metrics(
    save_returns_to,
    metric_names,
    metrics,
    agent_name,
    env_name,
    seed,
    config=None,
    last_eval_rewards=None,
    suptitle=None,
):
    now = time.time()
    new_dir = agent_name + f"-{env_name}-seed-{seed}-{now}"
    new_dir = save_returns_to / new_dir
    new_dir.mkdir(parents=True)
    save_returns_to = new_dir

    # save rewards from last eval episode if given;
    if last_eval_rewards is not None:
        file_name = save_returns_to / "last-eval-episode-rewards.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(last_eval_rewards, f)

    # save pickle files;
    for metric_name, metric in zip(metric_names, metrics):
        file_name = f"{metric_name}-seed-{seed}.pkl"
        file_name = save_returns_to / file_name
        with open(file_name, "wb") as f:
            pickle.dump(metric, f)
    if config is not None:
        file_name = save_returns_to / "config.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(config, f)

    # save plots of the metrics;
    save_metric_plots(
        metric_names,
        metrics,
        save_returns_to,
        seed,
        suptitle=suptitle,
    )
