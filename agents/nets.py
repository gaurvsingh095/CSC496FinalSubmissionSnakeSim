# agents/nets.py
import torch, torch.nn as nn

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x): return self.net(x.float())

class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.adv   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
    def forward(self, x):
        z = self.feature(x.float())
        v, a = self.value(z), self.adv(z)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q
