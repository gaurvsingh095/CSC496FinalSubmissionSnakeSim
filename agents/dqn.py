import torch, torch.nn as nn
import numpy as np, random
from collections import deque, namedtuple

Transition = namedtuple("T", "s a r s2 d")

class QNet(nn.Module):
    def __init__(self, in_ch, height, width, n_actions):
        super().__init__()
        # Small-grid friendly: 3x3 convs + adaptive pooling to 4x4
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flat = 64 * 4 * 4
        self.head = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        z = self.conv(x)
        z = self.pool(z)
        return self.head(z.view(z.size(0), -1))

class Replay:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)
    def __len__(self): return len(self.buf)
    def push(self, *t): self.buf.append(Transition(*t))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s  = np.stack([b.s  for b in batch], axis=0)
        a  = np.array([b.a  for b in batch], dtype=np.int64)
        r  = np.array([b.r  for b in batch], dtype=np.float32)
        s2 = np.stack([b.s2 for b in batch], axis=0)
        d  = np.array([b.d  for b in batch], dtype=np.float32)
        return s, a, r, s2, d
