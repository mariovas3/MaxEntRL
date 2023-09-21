import math

import numpy as np
import torch
from torch import nn

from rl_algos.algorithms.distributions import batch_UT_trick_from_samples
from rl_algos.algorithms.sac_base import SACAgentBase, TEST_OUTPUTS_PATH
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import time


class SACAgent(SACAgentBase):
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
        UT_trick=False,
        with_entropy=False,
        **kwargs,
    ):
        super(SACAgent, self).__init__(
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
            save_to,
            cache_best_policy,
            clip_grads,
            fixed_temperature,
            num_policy_updates,
            num_qfunc_updates,
            **kwargs,
        )
        self.UT_trick = UT_trick
        self.with_entropy = with_entropy

        # update name;
        self.name = (
            name + f"-{self.policy.name}-UT-{int(self.UT_trick)}"
            f"-entropy-{int(self.with_entropy)}"
            f"-buffer-size-{self.buffer.max_size}"
            f"-iters-{self.num_iters}"
            f"-env-{self.env.spec.id}"
            f"-seed-{self.seed}"
            f"-timestamp-{time.time()}"
        )

        # created save dir;
        self.save_to = save_to
        if self.save_to is not None:
            self.save_to = self.save_to / self.name
            if not self.save_to.exists():
                self.save_to.mkdir(parents=True)

    def sample_action(
        self,
        obs,
        give_density=False,
        k_proposals: int = 1,
    ):
        policy_dist = self.policy(obs)
        a = policy_dist.sample(k_proposals)
        if give_density:
            return a, policy_dist
        return a


    def sample_deterministic(self, obs, give_density=False):
        policy_dist = self.policy(obs)
        if give_density:
            return policy_dist.mean, policy_dist
        return policy_dist.mean

    def get_policy_loss_and_temperature_loss(self, obs_t):
        # freeze q-nets and eval at current observation
        # and reparam trick action and choose min for policy loss;
        self.Q1.requires_grad_(False)
        self.Q2.requires_grad_(False)
        self.Q1.eval()
        self.Q2.eval()
        self.policy.train()

        # get policy;
        policy_density = self.policy(obs_t)

        if self.UT_trick:
            # currently works for gaussian policy;
            if self.with_entropy:
                log_pi_integral = -policy_density.entropy().sum(-1)
            else:
                log_pi_integral = policy_density.log_prob_UT_trick().sum(
                    -1
                )
            UT_trick_samples = policy_density.get_UT_trick_input()

            # get the integrals;
            q1_integral = batch_UT_trick_from_samples(
                self.Q1.net,
                obs_t,
                UT_trick_samples,
            )
            q2_integral = batch_UT_trick_from_samples(
                self.Q2.net,
                obs_t,
                UT_trick_samples,
            )
            
            # get the lower estimate;
            q_integral = torch.min(q1_integral, q2_integral).view(-1)

            # get policy_loss;
            self.policy_loss = (
                math.exp(self.log_temperature.item()) * log_pi_integral
                - q_integral
            ).mean()

            # get temperature loss;
            if self.fixed_temperature is None:
                self.temperature_loss = -(
                    self.log_temperature.exp()
                    * (log_pi_integral + self.entropy_lb).detach()
                ).mean()
        else:
            # do reparam trick;
            repr_trick = policy_density.rsample().squeeze()
            assert repr_trick.requires_grad

            # get log prob for policy optimisation;
            if self.with_entropy:
                log_prob = -policy_density.entropy().sum(-1)
            else:
                log_prob = policy_density.log_prob(
                    repr_trick
                ).sum(-1)

            qfunc_in = torch.cat((obs_t, repr_trick), -1)
            q_est = torch.min(
                self.Q1.net(qfunc_in), self.Q2.net(qfunc_in)
            ).view(-1)

            # get loss for policy;
            self.policy_loss = (
                math.exp(self.log_temperature.item()) * log_prob - q_est
            ).mean()

            if self.fixed_temperature is None:
                # get temperature loss;
                self.temperature_loss = -(
                    self.log_temperature.exp()
                    * (log_prob + self.entropy_lb).detach()
                ).mean()

        # housekeeping;
        self.policy_losses.append(self.policy_loss.item())
        if self.fixed_temperature is None:
            self.temperature_losses.append(self.temperature_loss.item())
        self.policy.eval()

    def get_q_losses(
        self,
        obs_t,
        action_t,
        reward_t,
        obs_tp1,
        terminated_tp1,
    ):
        self.Q1.requires_grad_(True)
        self.Q2.requires_grad_(True)
        self.Q1.train()
        self.Q2.train()
        self.policy.eval()

        # get predictions from q functions;
        policy_density = self.policy(obs_tp1)
        obs_action_t = torch.cat((obs_t, action_t), -1)
        q1_est = self.Q1(
            obs_action_t
        ).view(-1)
        q2_est = self.Q2(
            obs_action_t
        ).view(-1)

        if self.UT_trick:
            # get (B, 2 * action_dim + 1, action_dim) samples;
            UT_trick_samples = policy_density.get_UT_trick_input()
            # eval expectation of q-target functions by averaging over the
            # 2 * action_dim + 1 samples and get (B, 1) output;
            qt1_est = batch_UT_trick_from_samples(
                self.Q1t.net,
                obs_tp1,
                UT_trick_samples,
            )
            qt2_est = batch_UT_trick_from_samples(
                self.Q2t.net,
                obs_tp1,
                UT_trick_samples,
            )
            # get negative entropy by using the UT trick;
            if self.with_entropy:
                log_probs = -policy_density.entropy().sum(-1)
            else:
                log_probs = policy_density.log_prob_UT_trick().sum(-1)
        else:
            # sample future action;
            action_tp1 = policy_density.sample().squeeze()

            # get log probs;
            if self.with_entropy:
                log_probs = -policy_density.entropy().sum(-1).view(-1)
            else:
                log_probs = (
                    policy_density.log_prob(action_tp1).sum(-1).view(-1)
                )

            obs_action_tp1 = torch.cat((obs_tp1, action_tp1), -1)
            qt1_est = self.Q1t.net(obs_action_tp1)
            qt2_est = self.Q2t.net(obs_action_tp1)

        # use the values from the target net that
        # had lower value predictions;
        q_target = (
            torch.min(qt1_est, qt2_est).view(-1)
            - self.log_temperature.exp() * log_probs
        )

        q_target = (
            reward_t
            + (1 - terminated_tp1.int()) * self.discount * q_target
        ).detach()

        # loss for first q func;
        self.Q1_loss = nn.MSELoss()(q1_est, q_target)

        # loss for second q func;
        self.Q2_loss = nn.MSELoss()(q2_est, q_target)

        self.Q1_losses.append(self.Q1_loss.item())
        self.Q2_losses.append(self.Q2_loss.item())

    def train_one_epoch(self):
        self.policy.eval()
        # presample if needed;
        self.check_presample()

        for _ in tqdm(range(self.num_iters)):
            # sample paths;
            self.policy.eval()
            self.buffer.collect_path(
                self.env,
                self,
                self.num_steps_to_sample,
            )

            # eval paths with policy;
            (
                r,
                log_probs,
                code,
                ep_len,
            ) = self.buffer.get_single_ep_rewards_and_logprobs(
                self.env,
                self,
            )
            assert ep_len == len(r)
            if isinstance(r, list):
                r = sum(r)
            else:
                r = r.sum().item()
            self.eval_path_returns.append(r)
            self.eval_path_avg_rewards.append(r / ep_len)
            self.eval_path_lens.append(ep_len)
            self.eval_path_avg_log_probs.append(np.mean(log_probs).item())

            if self.cache_best_policy:
                if (
                    self.best_eval_return is None
                    or self.eval_path_returns[-1] > self.best_eval_return
                ):
                    self.best_eval_return = self.eval_path_returns[-1]
                    self._cache()
            

            if (
                self.num_qfunc_updates is not None
                and self.num_policy_updates is not None
                and self.num_qfunc_updates != self.num_policy_updates
            ):
                self.do_k_gradsteps_qfuncs()
                self.do_k_gradsteps_policy_temp()
            else:
                self.do_k_gradsteps_together()
    
    def do_k_gradsteps_together(self):
        # do the gradient updates;
        for _ in range(self.num_grad_steps):
            (
                obs_t,
                action_t,
                reward_t,
                obs_tp1,
                terminated_tp1,
            ) = self.buffer.sample(self.batch_size)

            # qfunc losses;
            self.get_q_losses(
                obs_t,
                action_t,
                reward_t,
                obs_tp1,
                terminated_tp1,
            )

            # get temperature and policy loss;
            self.get_policy_loss_and_temperature_loss(
                obs_t
            )

            # grad step on qfuncs, policy and temp;
            self.update_qfuncs()
            self.update_policy_and_temp()

            # target q funcs update;
            self.track_qfunc_params()
    
    def do_k_gradsteps_qfuncs(self):
        for _ in range(self.num_qfunc_updates):
            (
                obs_t,
                action_t,
                reward_t,
                obs_tp1,
                terminated_tp1,
            ) = self.buffer.sample(self.batch_size)

            # qfunc losses;
            self.get_q_losses(
                obs_t,
                action_t,
                reward_t,
                obs_tp1,
                terminated_tp1,
            )

            # grad step on qfuncs;
            self.update_qfuncs()

            # target q funcs update;
            self.track_qfunc_params()
    
    def do_k_gradsteps_policy_temp(self):
        for _ in range(self.num_policy_updates):
            (
                obs_t,
                _,
                _,
                _,
                _,
            ) = self.buffer.sample(self.batch_size)

            # get temperature and policy loss;
            self.get_policy_loss_and_temperature_loss(
                obs_t
            )

            # grad step on policy and temp;
            self.update_policy_and_temp()
