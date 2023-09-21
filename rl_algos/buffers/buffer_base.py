import numpy as np
np.random.seed(0)


class BufferBase:
    def __init__(
        self,
        max_size,
        seed=None,
    ):
        self.seed = 0 if seed is None else seed
        self.idx, self.max_size = 0, max_size
        self.looped = False

        # log variables;
        self.undiscounted_returns = []
        self.path_lens = []
        self.avg_reward_per_episode = []

    def add_sample(self, *args, **kwargs):
        pass

    def sample(self, batch_size):
        pass

    def __len__(self):
        return self.max_size if self.looped else self.idx

    def collect_path(self, env, agent, num_steps_to_collect):
        pass

    def clear_buffer(self):
        """Reset tracked logs and idx in buffer."""
        self.idx = 0
        self.looped = False

        self.undiscounted_returns = []
        self.path_lens = []
        self.avg_reward_per_episode = []
