# Snake RL — DQN + Visuals (Class-Ready)

This repo contains:
- **Gymnasium-compatible Snake environment** (pixel & feature modes)
- **Small-grid-safe DQN** (adaptive pooling; works on 8–20+ boards)
- **Baselines** (Greedy + Tabular Q)
- **Live viewer (Pygame)** and **GIF recorder**

## Quick Start
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Train DQN
```bash
python train_dqn.py                  # default: size=10, steps=300k
# or
python train_dqn.py --size 12 --frames 4 --steps 500000
```
TensorBoard:
```bash
tensorboard --logdir runs
```

### Evaluate
```bash
python eval.py --ckpt checkpoints/dqn_latest.pt --episodes 20 --size 10
```

### Watch it play (live window)
```bash
python visualize/play_dqn.py --ckpt checkpoints/dqn_latest.pt --size 10 --fps 12 --scale 24
```

### Export a GIF
```bash
python visualize/record_gif.py --ckpt checkpoints/dqn_latest.pt --size 10 --episodes 3 --out snake_play.gif
```

### Tabular Baseline (feature state)
```bash
python train_tabular.py --size 8 --episodes 20000
```

## Repo Layout
```
envs/
  snake_env.py
agents/
  dqn.py
baselines/
  greedy.py
  tabular_q.py
visualize/
  play_dqn.py
  record_gif.py
train_dqn.py
train_tabular.py
eval.py
configs/dqn_default.yaml
requirements.txt
README.md
```

## Notes
- DQN uses replay buffer, target network, Huber loss, ε-greedy.
- `agents/dqn.py` uses **3×3 convs + AdaptiveAvgPool2d(4×4)** so it runs on small boards.
- `envs/snake_env.py` exposes `render_array(scale)` for RGB frames.
