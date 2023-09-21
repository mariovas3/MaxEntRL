from rl_algos.buffers.buffer_base import BufferBase
import numpy as np
import torch


class GymBuffer(BufferBase):
    def __init__(
        self,
        max_size,
        obs_dim,
        action_dim,
        seed=None,
        reward_scale=1.0,
        verbose=False,
        be_deterministic=False,
    ):
        super(GymBuffer, self).__init__(
            max_size,
            seed,
        )
        
        self.verbose = verbose
        self.be_deterministic = be_deterministic

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # init storage containers;
        self.obs_t = np.empty(
            (max_size, obs_dim), dtype=np.float32
        )
        self.action_t = np.empty(
            (max_size, action_dim), dtype=np.float32
        )
        self.reward_t = np.empty(
            (max_size,), dtype=np.float32
        )
        self.obs_tp1 = np.empty(
            (max_size, obs_dim), dtype=np.float32
        )
        self.terminal_tp1 = np.empty((max_size,), dtype=np.int8)

        # the reward scale has the role of balancing the 
        # RL objective, low reward_scale leads to high-entropy 
        # seeking behaviour; high reward_scale, means we prioritise 
        # maximising the reward;
        # this was important in the first version of SAC
        # but in the second they introduced 
        # auto-tuning of the temperature param so the reward scale 
        # should be less important;
        self.reward_scale = reward_scale

    def clear_buffer(self):
        super().clear_buffer()
        # init storage containers;
        self.obs_t = np.empty(
            (self.max_size, self.obs_dim), dtype=np.float32
        )
        self.action_t = np.empty(
            (self.max_size, self.action_dim), dtype=np.float32
        )
        self.reward_t = np.empty(
            (self.max_size,), dtype=np.float32
        )
        self.obs_tp1 = np.empty(
            (self.max_size, self.obs_dim), dtype=np.float32
        )
        self.terminal_tp1 = np.empty((self.max_size,), dtype=np.int8)

    def add_sample(self, obs_t, action_t, reward_t, obs_tp1, terminal_tp1):
        idx = self.idx % self.max_size
        if not self.looped and self.idx and not idx:
            self.looped = True
        self.idx = idx
        self.obs_t[idx] = obs_t
        self.action_t[idx] = action_t
        self.reward_t[idx] = reward_t
        self.obs_tp1[idx] = obs_tp1
        self.terminal_tp1[idx] = terminal_tp1
        self.idx += 1

    def sample(self, batch_size):
        assert batch_size <= self.__len__()

        # always sample without replacement;
        idxs = np.random.choice(
            self.__len__(), size=batch_size, replace=False
        )

        return (
            torch.from_numpy(self.obs_t[idxs]),
            torch.from_numpy(self.action_t[idxs]),
            torch.from_numpy(self.reward_t[idxs]),
            torch.from_numpy(self.obs_tp1[idxs]),
            torch.tensor(self.terminal_tp1[idxs], dtype=torch.float32),
        )

    def get_single_ep_rewards_and_logprobs(
        self,
        env,
        agent,
    ):
        reward_on_path = []
        log_probs_on_path = []

        # start the episode;
        obs, info = env.reset(seed=self.seed)
        terminated, truncated = False, False

        # this steps variable is for debugging
        steps = 0

        # process the episode;
        while not (terminated or truncated):
            # sample actions;
            if self.be_deterministic:
                a, policy_dist = agent.sample_deterministic(
                    torch.tensor(obs, dtype=torch.float32),
                    give_density=True
                )
                a = a.detach()
            else:
                a, policy_dist = agent.sample_action(
                    torch.tensor(obs, dtype=torch.float32),
                    give_density=True
                )
            # flatten action;
            a = a.view(-1)

            # make env step;
            new_obs, reward, terminated, truncated, info = env.step(a.numpy())

            # track rewards on path;
            reward_on_path.append(reward)
            log_probs_on_path.append(policy_dist.log_prob(a).sum(-1).item())
            # if self.verbose:
                # print(
                    #   f"reward: {reward_on_path[-1]}\n"
                    #   f"log-prob: {log_probs_on_path[-1].item()}"
                    # )
            # update current state;
            obs = new_obs
            steps += 1
            code = -1
            if terminated and not truncated:
                code = 0
            elif terminated and truncated:
                code = 1
            elif truncated:
                code = 2
            if code != -1:
                if self.verbose:
                    print(f"steps done: {steps}, code: {code}")
                return (
                    reward_on_path,
                    log_probs_on_path,
                    code,
                    steps,
                )

        if self.verbose:
            print(f"steps done: {steps}, code: {code}")
        return (
            reward_on_path,
            log_probs_on_path,
            2,  # truncated;
            steps,
        )

    def collect_path(
        self,
        env,
        agent,
        num_steps_to_collect,
    ):
        """
        Collect experience from MDP.

        Args:
            env: Supports similar api to gymnasium.Env..
            agent: Supports sample_action(obs) api.
            num_steps_to_collect: Number of (obs, action, reward, next_obs, terminated)
                tuples to be added to the buffer.

        Note:
            This will collect (obs_t, action_t, reward, obs_tp1, terminal)
            tuples as steps.
        """
        num_steps_to_collect = min(num_steps_to_collect, self.max_size)
        t = 0
        obs_t, info = env.reset(seed=self.seed)
        self.seed += 1

        num_rewards, avg_reward, undiscounted_return = 0, 0., 0.

        # get at least num_steps_to_collect steps
        # and exit when terminated or truncated;
        while t < num_steps_to_collect or not (terminated or truncated):
            # sample action;
            a = agent.sample_action(
                torch.tensor(obs_t, dtype=torch.float32)
            ).view(-1).numpy()

            # sample dynamics;
            obs_tp1, reward, terminated, truncated, info = env.step(a)

            # house keeping for observed rewards.
            num_rewards += 1
            avg_reward = avg_reward + (reward - avg_reward) / num_rewards
            undiscounted_return += reward

            # add tuple;
            assert obs_t.ndim == a.ndim == 1
            self.add_sample(
                obs_t, a, reward, obs_tp1, terminated
            )

            # restart env if episode ended;
            if terminated or truncated:
                obs_t, info = env.reset(seed=self.seed)
                self.seed += 1
                self.undiscounted_returns.append(undiscounted_return)
                self.avg_reward_per_episode.append(avg_reward)
                self.path_lens.append(num_rewards)
                avg_reward = 0.0
                num_rewards = 0.0
                undiscounted_return = 0.0
            else:
                obs_t = obs_tp1
            t += 1


def sample_eval_path(T, env, agent, seed):
    agent.policy.eval()
    observations, actions, rewards = [], [], []
    obs, info = env.reset(seed=seed)
    observations.append(obs)
    for _ in range(T):
        obs = torch.tensor(obs, dtype=torch.float32)
        action = agent.sample_deterministic(obs).numpy()
        new_obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        observations.append(new_obs)
        rewards.append(reward)
        obs = new_obs
        if terminated and not truncated:
            return observations, actions, rewards, 0
        if terminated and truncated:
            return observations, actions, rewards, 1
        if truncated:
            return observations, actions, rewards, 2
    return observations, actions, rewards, 2


if __name__ == "__main__":
    import gymnasium as gym

    class DummyAgent:
        def __init__(self, env):
            self.env = env

        def sample_action(self, obs_t):
            return torch.tensor(
                self.env.action_space.sample(), dtype=torch.float32
            )

    env = gym.make("Hopper-v2", max_episode_steps=300)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_size = 10_000
    batch_size = 100
    buffer = GymBuffer(max_size, obs_dim, action_dim)
    returns = []
    agent = DummyAgent(env)
    buffer.collect_path(env, agent, 1200)
    print(len(buffer))
    for _ in range(5):
        batch = buffer.sample(batch_size)
        print(f"len(batch): {len(batch[0])}")
