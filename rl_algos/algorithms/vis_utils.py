import numpy as np
import matplotlib.pyplot as plt
import pygame
from math import sqrt
import time
import torch
import pickle


def plot_overlayed_hists(
    entity_names,
    metric_names,
    *metrics_for_all,
    figsize=(14, 8),
    path_to_save=None,
):
    fig = plt.figure(figsize=figsize)
    for i, n in enumerate(metric_names):
        plt.subplot(1, len(metric_names), i + 1)
        ax = plt.gca()
        for j, metrics_for_one in enumerate(metrics_for_all):
            plt.hist(
                metrics_for_one[i],
                log=True,
                alpha=0.5,
                label=entity_names[j],
            )
        plt.legend()
        plt.xlabel(n)
    fig.tight_layout()
    if path_to_save is not None:
        plt.savefig(path_to_save)
    else:
        plt.show()


def see_one_episode(env, agent, seed, save_to):
    save_to = save_to / 'animation-episode-obs-actions.pkl'
    obs, info = env.reset(seed=seed)
    done = False
    step = 0
    obs_list = [obs]
    action_list = []
    while not done:
        time.sleep(1 / 60)  # slow down rendering, otherwise 125 fps;
        obs = torch.tensor(obs, dtype=torch.float32)
        action = agent.sample_deterministic(obs).detach().numpy()
        action_list.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_list.append(obs)
        done = terminated or truncated
        step += 1
        if terminated:
            print(f"simulation died step: {step}")
    # save episode;
    with open(save_to, "wb") as f:
        pickle.dump(obs_list, f)
        pickle.dump(action_list, f)
    env.close()
    pygame.display.quit()


def save_metric_plots(metric_names, metrics, path, seed, suptitle=None):
    file_name = f"metric-plots-seed-{seed}.png"
    file_name = path / file_name
    rows = int(sqrt(len(metrics)))
    cols = int(len(metrics) / rows) + (1 if len(metrics) % rows else 0)
    fig = plt.figure(figsize=(12, 8))
    for i, (metric_name, metric) in enumerate(
        sorted(zip(metric_names, metrics), key=lambda x: x[0])
    ):
        plt.subplot(rows, cols, i + 1)
        ax = plt.gca()
        ax.plot(metric)
        try:
            ylow, yhigh = np.min(metric), np.max(metric)
            yoffset = max((yhigh - ylow) / 10, 0.1)
            ax.set_ylim(ylow, yhigh)
            ax.set_yticks(
                np.arange(ylow - 1.5 * yoffset, yhigh + 1.5 * yoffset, yoffset)
            )   
        except:
            print(f"problem with {metric_name}\n", metric)
        
        xlow, xhigh = 0, len(metric) + 5
        ax.set_ylabel(metric_name)
        ax.set_xticks(np.arange(xlow, xhigh, (xhigh - xlow) / 10).round())
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    plt.savefig(file_name)
    plt.close()


def visualise_episode(env, agent, seed, T=None):
    """
    Assumes env has "human" in env.metadata["render_modes"].
    """
    obs, info = env.reset(seed=seed)
    done = False
    t = 0
    pred = t < T if T else True
    while not done and pred:
        action = agent.sample_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if T:
            t += 1
            pred = t < T
        done = terminated or truncated
    pygame.display.quit()


def get_moving_avgs(returns_over_episodes, num_steps):
    """
    Return len(returns_over_episodes) - num_steps + 1 moving
    averages over num_steps steps.
    """
    if num_steps > len(returns_over_episodes):
        print(
            "issue in moving avg generation; "
            f"not enough data for {num_steps} step ma;\n"
        )
        return []
    return (
        np.correlate(
            returns_over_episodes, np.ones(num_steps), mode="valid"
        )
        / num_steps
    )


def ci_plot(ax, data: np.ndarray, **kwargs):
    """
    Plot mean +- std intervals along the rows of data.
    """
    x = np.arange(data.shape[-1])
    avg = np.mean(data, 0)
    sd = np.std(data, 0)
    ax.fill_between(x, avg - sd, avg + sd, alpha=0.2, **kwargs)
    ax.plot(x, avg, **kwargs)
