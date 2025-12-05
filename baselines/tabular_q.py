import numpy as np
from collections import defaultdict
import random

class TabularQ:
    """Feature-state tabular Q for small boards."""
    def __init__(self, n_actions=4, alpha=0.2, gamma=0.99, eps=1.0, eps_min=0.05, eps_decay=0.9995):
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
        self.n_actions = n_actions

    def key(self, obs):
        x = np.clip(obs, -1, 1)
        bins = np.round(x*5).astype(int)  # coarse discretization
        return tuple(bins.tolist())

    def act(self, obs):
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        q = self.Q[self.key(obs)]
        return int(np.argmax(q))

    def step(self, s, a, r, s2, done):
        q = self.Q[self.key(s)]
        qn = self.Q[self.key(s2)]
        target = r + (0 if done else self.gamma*np.max(qn))
        q[a] += self.alpha*(target - q[a])
        self.eps = max(self.eps_min, self.eps*self.eps_decay)
