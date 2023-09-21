import torch
import re
import gymnasium as gym

from rl_algos.algorithms.policy import *
from rl_algos.buffers.gym_buffers import GymBuffer

import warnings


params_func_config = dict(
    env_name=None,
    policy_epochs=1,
    num_iters=100,
    batch_size=100,
    num_grad_steps=1,
    num_steps_to_sample=1000,
    reward_scale=1.0,
    net_hiddens=[64],
    UT_trick=False,
    ortho_init=False,
    seed=0,
    clip_grads=False,
    fixed_temperature=None,
    max_size=10_000,
    with_mlp_batch_norm=False,
    with_mlp_layer_norm=True,
    policy_lr=1e-3,
    temperature_lr=1e-3,
    qfunc_lr=1e-3,
    log_sigma_min=None,  # -20,
    log_sigma_max=None,  # 2,
    num_policy_updates=None,
    num_qfunc_updates=None,
)


def get_params(
    env_name,
    policy_epochs=1,
    num_iters=100,
    batch_size=100,
    num_grad_steps=1,
    num_steps_to_sample=1000,
    reward_scale=1.0,
    net_hiddens=[64],
    UT_trick=False,
    ortho_init=False,
    seed=0,
    clip_grads=False,
    fixed_temperature=None,
    max_size=10_000,
    with_mlp_batch_norm=False,
    with_mlp_layer_norm=True,
    policy_lr=1e-3,
    temperature_lr=1e-3,
    qfunc_lr=1e-3,
    log_sigma_min=None,  # -20,
    log_sigma_max=None,  # 2,
    num_policy_updates=None,
    num_qfunc_updates=None,
):
    assert env_name
    if isinstance(net_hiddens, int):
        net_hiddens = [net_hiddens]
    gauss_policy_hiddens = net_hiddens
    qfunc_hiddens = net_hiddens
    which_policy_kwargs = "tanh_gauss_policy_kwargs"
    # which_policy_kwargs = 'tanhrange_beta_policy_kwargs'

    policy_constructors = dict(
        gauss_policy_kwargs=GaussPolicy,
        tanh_gauss_policy_kwargs=TanhGaussPolicy,
        tanhrange_beta_policy_kwargs=TanhRangeBetaPolicy,
    )

    env = gym.make(env_name)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

    policy_kwargs = dict(
        gauss_policy_kwargs=dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hiddens=gauss_policy_hiddens,
            with_batch_norm=with_mlp_batch_norm,
            with_layer_norm=with_mlp_layer_norm,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
        ),
        tanh_gauss_policy_kwargs=dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hiddens=gauss_policy_hiddens,
            with_batch_norm=with_mlp_batch_norm,
            with_layer_norm=with_mlp_layer_norm,
            log_sigma_min=log_sigma_min,
            log_sigma_max=log_sigma_max,
        ),
        tanhrange_beta_policy_kwargs=dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hiddens=gauss_policy_hiddens,
            with_layer_norm=with_mlp_layer_norm,
            with_batch_norm=with_mlp_batch_norm,
        )
    )

    if UT_trick and with_mlp_batch_norm:
        warnings.warn(
            "Qfunc cannot implement UT trick with batch norm on. "
            "This is because of the extra dimension created for "
            "the sigma points. BN will start treating this as the "
            "channel dimension which may be different (and usually is) "
            "from the intended dim of the BN layers in the MLP."
        )

    qfunc_kwargs = dict(
        obs_action_dim=obs_dim + action_dim,
        hiddens=qfunc_hiddens,
        with_layer_norm=with_mlp_layer_norm,
        # cant do UT trick with batch norm since batch_norm will
        # need to be applied to 2d + 1 nums, but it is init to handle
        # the output of the previous affine layer - so an error for
        # the dim will be raised;
        with_batch_norm=False if UT_trick else with_mlp_batch_norm,
    )

    Q1_kwargs = qfunc_kwargs.copy()
    Q2_kwargs = qfunc_kwargs.copy()
    Q1t_kwargs = qfunc_kwargs.copy()
    Q2t_kwargs = qfunc_kwargs.copy()

    agent_name = "SACAgent"
    agent_name = agent_name + (
        f"-nh-{len(net_hiddens)}x{net_hiddens[0]}"
    )
    agent_kwargs = dict(
        name=agent_name,
        policy_constructor=policy_constructors[which_policy_kwargs],
        qfunc_constructor=Qfunc,
        env=env,
        buffer_constructor=GymBuffer,
        optimiser_constructors=dict(
            policy_optim=torch.optim.Adam,
            temperature_optim=torch.optim.Adam,
            Q1_optim=torch.optim.Adam,
            Q2_optim=torch.optim.Adam,
        ),
        entropy_lb=action_dim * 2,
        policy_lr=policy_lr,
        temperature_lr=temperature_lr,
        qfunc_lr=qfunc_lr,
        tau=0.005,
        discount=1.0,
        save_to=TEST_OUTPUTS_PATH,
        cache_best_policy=False,
        clip_grads=clip_grads,
        fixed_temperature=fixed_temperature,
        UT_trick=UT_trick,
        with_entropy=False,
        num_policy_updates=num_policy_updates,
        num_qfunc_updates=num_qfunc_updates,
    )

    config = dict(
        training_kwargs=dict(
            seed=seed,
            num_iters=num_iters,
            num_steps_to_sample=num_steps_to_sample,
            num_grad_steps=num_grad_steps,
            num_policy_updates=num_policy_updates,
            num_qfunc_updates=num_qfunc_updates,
            batch_size=batch_size,
            num_eval_steps_to_sample=1500,
            min_steps_to_presample=max(300, 1500),
        ),
        Q1_kwargs=Q1_kwargs,
        Q2_kwargs=Q2_kwargs,
        Q1t_kwargs=Q1t_kwargs,
        Q2t_kwargs=Q2t_kwargs,
        policy_kwargs=policy_kwargs[which_policy_kwargs],
        buffer_kwargs=dict(
            max_size=max_size,
            seed=seed,
            obs_dim=obs_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            verbose=False,
            be_deterministic=False,
        ),
    )

    return agent_kwargs, config


def arg_parser(settable_params, argv):
    print(
        "\n------------------: USAGE :------------------\n"
        f"\nPass kwargs as name=value.\n"
        "If name is valid, value will be set.\n"
        "Valid kwargs to the script are:\n",
        settable_params.keys(),
        end="\n\n",
    )
    int_list_regex = re.compile("^([0-9]+,)+[0-9]+$")
    int_regex = re.compile("^(-?)[0-9]+[0-9]*$")
    if len(argv) > 1:
        for a in argv[1:]:
            n, v = a.split("=")
            if n in settable_params:
                if re.match(int_list_regex, v):
                    nums = v.split(",")
                    v = [int(temp) for temp in nums]
                elif re.match(int_regex, v):
                    v = int(v)
                elif "_coef" in n or "_lr" in n:
                    v = float(v)
                settable_params[n] = v
                print(f"{n}={v}", type(v), v)
            else:
                print(f"{n} not a valid argument")
    print("\n")
    return settable_params
