import torch
from torch import nn
import torch.distributions as dists
from pathlib import Path
from rl_algos.algorithms.distributions import *
from math import exp


TEST_OUTPUTS_PATH = Path(__file__).absolute().parent.parent.parent / "test_output"
if not TEST_OUTPUTS_PATH.exists():
    TEST_OUTPUTS_PATH.mkdir()


class AmortisedBetaNet(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hiddens,
        with_layer_norm=False,
        with_batch_norm=False,
    ):
        super(AmortisedBetaNet, self).__init__()
        assert not (with_batch_norm and with_layer_norm)
        
        # init net;
        self.net = nn.Sequential()

        # add modules/Layers to net;
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(obs_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i - 1], hiddens[i]))

            if with_batch_norm:
                self.net.append(nn.BatchNorm1d(hiddens[i]))
            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))
            # ReLU activation;
            self.net.append(nn.ReLU())

        # get a and b;
        self.a_net = nn.Sequential(
            nn.Linear(hiddens[-1], action_dim), nn.Softplus()
        )
        self.b_net = nn.Sequential(
            nn.Linear(hiddens[-1], action_dim), nn.Softplus()
        )
    
    def forward(self, obs):
        emb = self.net(obs)
        return self.a_net(emb), self.b_net(emb)


class TanhRangeBetaPolicy(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hiddens,
        with_layer_norm=False,
        with_batch_norm=False,
    ):
        super(TanhRangeBetaPolicy, self).__init__()
        self.name = "TanhRangeBetaPolicy"
        self.net = AmortisedBetaNet(
            obs_dim,
            action_dim,
            hiddens,
            with_layer_norm=with_layer_norm,
            with_batch_norm=with_batch_norm,
        )
    
    def forward(self, obs):
        a, b = self.net(obs)
        return TanhRangeBeta(dists.Beta(a, b))


class AmortisedGaussNet(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hiddens,
        with_layer_norm=False,
        with_batch_norm=False,
        log_sigma_min=None,
        log_sigma_max=None,
    ):
        super(AmortisedGaussNet, self).__init__()
        assert not (with_layer_norm and with_batch_norm)
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        # init net;
        self.net = nn.Sequential()

        # add modules/Layers to net;
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(obs_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i - 1], hiddens[i]))

            if with_batch_norm:
                self.net.append(nn.BatchNorm1d(hiddens[i]))
            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))
            # ReLU activation;
            self.net.append(nn.ReLU())

        # add mean and Cholesky of diag covariance net;
        self.mu_net = nn.Linear(hiddens[-1], action_dim)
        self.std_net = nn.Sequential(
            nn.Linear(hiddens[-1], action_dim), nn.Softplus()
        )

    def forward(self, obs):
        # assert not torch.any(obs.isnan())
        emb = self.net(obs)  # shared embedding for mean and std;
        if (
            self.log_sigma_max is not None
            and self.log_sigma_min is not None
        ):
            return (
                self.mu_net(emb),
                torch.clamp(
                    self.std_net(emb),
                    exp(self.log_sigma_min),
                    exp(self.log_sigma_max),
                ),
            )
        return self.mu_net(emb), self.std_net(emb)


class GaussPolicy(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hiddens,
        with_layer_norm=False,
        with_batch_norm=False,
        log_sigma_min=None,
        log_sigma_max=None,
    ):
        super(GaussPolicy, self).__init__()

        self.name = "GaussPolicy"

        # init net;
        self.net = AmortisedGaussNet(
            obs_dim,
            action_dim,
            hiddens,
            with_layer_norm=with_layer_norm,
            with_batch_norm=with_batch_norm,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
        )

    def forward(self, obs):
        mus, sigmas = self.net(obs)
        return GaussDist(dists.Normal(mus, sigmas))


class TanhGaussPolicy(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hiddens,
        with_layer_norm=False,
        with_batch_norm=False,
        log_sigma_min=None,
        log_sigma_max=None,
    ):
        super(TanhGaussPolicy, self).__init__()
        self.name = "TanhGaussPolicy"

        self.net = AmortisedGaussNet(
            obs_dim,
            action_dim,
            hiddens,
            with_layer_norm=with_layer_norm,
            with_batch_norm=with_batch_norm,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
        )

    def forward(self, obs):
        mus, sigmas = self.net(obs)
        return TanhGauss(dists.Normal(mus, sigmas))


class Qfunc(nn.Module):
    def __init__(
        self,
        obs_action_dim,
        hiddens,
        with_layer_norm=False,
        with_batch_norm=False,
    ):
        super(Qfunc, self).__init__()

        self.with_batch_norm = with_batch_norm
        self.with_layer_norm = with_layer_norm
        assert not (with_batch_norm and with_layer_norm)

        # init net;
        self.net = nn.Sequential()

        # add hidden layers;
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(obs_action_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i - 1], hiddens[i]))

            if with_batch_norm:
                self.net.append(nn.BatchNorm1d(hiddens[i]))
            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))
            # add ReLU non-linearity;
            self.net.append(nn.ReLU())

        # Q-func maps to scalar;
        self.net.append(nn.Linear(hiddens[-1], 1))

    def forward(self, obs_action):
        return self.net(obs_action)
