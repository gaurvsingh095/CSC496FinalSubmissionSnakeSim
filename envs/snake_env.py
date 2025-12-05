import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random

# Directions: 0=up,1=right,2=down,3=left
DIRS = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}

class SnakeEnv(gym.Env):
    """
    Snake with pixel or feature observations and simple RGB rendering.
    """
    metadata = {"render_modes": []}

    def __init__(self, size=10, feature_state=False, frame_stack=1, step_penalty=-0.01, seed=None):
        super().__init__()
        assert size >= 5, "Use size>=5"
        self.size = size
        self.feature_state = feature_state
        self.frame_stack = frame_stack
        self.step_penalty = float(step_penalty)
        self.rng = np.random.RandomState(seed)
        self.action_space = spaces.Discrete(4)

        if feature_state:
            # distances-to-walls (4), apple vec (2), dir onehot (4), dangers L/S/R (3)
            self.obs_dim = 4 + 2 + 4 + 3
            self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_dim*frame_stack,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(3*frame_stack, size, size), dtype=np.float32)

        self.snake = None
        self.direction = None
        self.apple = None
        self.frames = deque(maxlen=frame_stack)
        self.step_count = 0
        self.max_steps = size*size*10  # safety cap

    # ---------- setup helpers ----------
    def _spawn_apple(self):
        free = [(r,c) for r in range(self.size) for c in range(self.size) if (r,c) not in self.snake]
        self.apple = random.choice(free)

    def _collides(self, head):
        r,c = head
        if r < 0 or c < 0 or r >= self.size or c >= self.size:
            return True
        return head in list(self.snake)[:-1]

    def _place_initials(self):
        mid = self.size//2
        self.snake = deque([(mid, mid), (mid, mid-1), (mid, mid-2)])
        self.direction = 1  # right
        self._spawn_apple()
        self.step_count = 0

    # ---------- observations ----------
    def _grid_channels(self):
        head = np.zeros((self.size, self.size), np.float32)
        body = np.zeros_like(head)
        food = np.zeros_like(head)
        (hr,hc) = self.snake[0]
        head[hr, hc] = 1.0
        for r,c in list(self.snake)[1:]:
            body[r, c] = 1.0
        fr,fc = self.apple
        food[fr, fc] = 1.0
        return np.stack([head, body, food], axis=0)

    def _features(self):
        (hr,hc) = self.snake[0]
        # normalized distances to walls: up,right,down,left
        up, right, down, left = hr, self.size-1-hc, self.size-1-hr, hc
        v = np.array([up, right, down, left], dtype=np.float32)/(self.size-1)
        dx = (self.apple[1] - hc) / max(1, (self.size-1))
        dy = (self.apple[0] - hr) / max(1, (self.size-1))
        dir_onehot = np.eye(4, dtype=np.float32)[self.direction]

        def danger(dir_idx):
            dr, dc = DIRS[dir_idx]
            nh = (hr+dr, hc+dc)
            return 1.0 if self._collides(nh) else 0.0

        left_turn = (self.direction + 3) % 4
        right_turn = (self.direction + 1) % 4
        forward = self.direction
        dangers = np.array([danger(left_turn), danger(forward), danger(right_turn)], dtype=np.float32)
        return np.concatenate([v, [dx, dy], dir_onehot.astype(np.float32), dangers], axis=0)

    def _obs(self):
        if self.feature_state:
            x = self._features().astype(np.float32)
            while len(self.frames) < self.frame_stack:
                self.frames.append(np.zeros_like(x))
            self.frames.append(x)
            return np.concatenate(list(self.frames), axis=0)
        else:
            x = self._grid_channels().astype(np.float32)
            while len(self.frames) < self.frame_stack:
                self.frames.append(np.zeros_like(x))
            self.frames.append(x)
            return np.concatenate(list(self.frames), axis=0)

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            self.rng = np.random.RandomState(seed)
        self.frames.clear()
        self._place_initials()
        obs = self._obs()
        return obs, {}

    def step(self, action):
        if (action + 2) % 4 == self.direction:
            action = self.direction
        self.direction = int(action)

        dr, dc = DIRS[self.direction]
        head = self.snake[0]
        new_head = (head[0] + dr, head[1] + dc)
        self.step_count += 1

        reward = self.step_penalty
        terminated = False

        if self._collides(new_head) or self.step_count >= self.max_steps:
            reward = -1.0
            terminated = True
        else:
            self.snake.appendleft(new_head)
            if new_head == self.apple:
                reward = 1.0
                self._spawn_apple()
            else:
                self.snake.pop()

        obs = self._obs()
        return obs, reward, terminated, False, {}

    # ---------- simple renders ----------
    def render_ascii(self):
        grid = [["." for _ in range(self.size)] for __ in range(self.size)]
        for r,c in list(self.snake)[1:]:
            grid[r][c] = "o"
        hr,hc = self.snake[0]
        grid[hr][hc] = "H"
        fr,fc = self.apple
        grid[fr][fc] = "A"
        return "\n".join(" ".join(row) for row in grid)

    def render_array(self, scale=16):
        """Return an RGB uint8 image array of the board at desired pixel scale."""
        h = w = self.size*scale
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # colors
        bg = np.array([24, 26, 34], dtype=np.uint8)
        grid_c = np.array([40, 44, 52], dtype=np.uint8)
        apple_c = np.array([220, 70, 70], dtype=np.uint8)
        head_c = np.array([80, 200, 120], dtype=np.uint8)
        body_c = np.array([60, 160, 100], dtype=np.uint8)

        img[:,:,:] = bg
        # grid
        for r in range(self.size):
            for c in range(self.size):
                y0,y1 = r*scale, (r+1)*scale
                x0,x1 = c*scale, (c+1)*scale
                img[y0:y1, x0:x1] = (img[y0:y1, x0:x1]*0.9 + grid_c*0.1).astype(np.uint8)

        # apple
        ar, ac = self.apple
        y0,y1 = ar*scale, (ar+1)*scale
        x0,x1 = ac*scale, (ac+1)*scale
        img[y0:y1, x0:x1] = apple_c

        # body
        for r,c in list(self.snake)[1:]:
            y0,y1 = r*scale, (r+1)*scale
            x0,x1 = c*scale, (c+1)*scale
            img[y0:y1, x0:x1] = body_c

        # head
        hr,hc = self.snake[0]
        y0,y1 = hr*scale, (hr+1)*scale
        x0,x1 = hc*scale, (hc+1)*scale
        img[y0:y1, x0:x1] = head_c

        return img
