# agents/replay.py
import random, math
import numpy as np

class ReplayBuffer:
    def __init__(self, cap):
        self.cap, self.ptr, self.size = cap, 0, 0
        self.s = np.zeros((cap, ), dtype=object)  # store tuples (s,a,r,s2,done)
    def add(self, exp):
        self.s[self.ptr] = exp
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
    def sample(self, batch):
        idx = np.random.randint(0, self.size, size=batch)
        return [self.s[i] for i in idx], idx
    def __len__(self): return self.size

class PERBuffer(ReplayBuffer):
    def __init__(self, cap, alpha=0.6, beta0=0.4):
        super().__init__(cap)
        self.alpha, self.beta0 = alpha, beta0
        self.prior = np.ones(cap, dtype=np.float32)
        self.t = 0
    def add(self, exp, priority=1.0):
        super().add(exp)
        self.prior[(self.ptr - 1) % self.cap] = max(priority, 1e-3)
    def sample(self, batch):
        N = self.size
        p = self.prior[:N] ** self.alpha
        p /= p.sum()
        idx = np.random.choice(N, size=batch, p=p)
        beta = min(1.0, self.beta0 + 0.4 * (self.t / 1e6))
        self.t += 1
        w = ((N * p[idx]) ** (-beta)).astype(np.float32)
        w /= w.max()
        return [self.s[i] for i in idx], idx, w
    def update_priorities(self, idx, td_err):
        self.prior[idx] = np.abs(td_err) + 1e-3
