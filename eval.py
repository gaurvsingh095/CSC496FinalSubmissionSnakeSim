# video_recorder.py
import yaml, torch, imageio.v2 as imageio
from pathlib import Path
from envs.snake_env import SnakeEnv
from agents.nets import QNet, DuelingQNet
import numpy as np

def _load_cfg(config_path: str):
    p = Path(config_path)
    text = p.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text) or {}
    if "env" in cfg:
        env_cfg = cfg["env"] or {}
    else:
        env_cfg = {
            "board_size":   cfg.get("board_size", 8),
            "feature_state":cfg.get("feature_state", True),
            "frame_stack":  cfg.get("frame_stack", 1),
            "step_penalty": cfg.get("step_penalty", -0.003),
        }
    return cfg, env_cfg

def _obs_dim_from_env(env, obs):
    if hasattr(env, "obs_dim"):
        return env.obs_dim
    if obs.ndim == 1:
        return obs.shape[0]
    return int(np.prod(obs.shape))

def record(ckpt, config_path, out="submission\\snake_demo.mp4", steps=1000, dueling=False, seed=123):
    cfg, env_cfg = _load_cfg(config_path)

    env = SnakeEnv(**env_cfg, seed=seed, render_mode="rgb_array")
    obs,_ = env.reset(seed=seed)
    obs_dim = _obs_dim_from_env(env, obs)
    n_actions = env.n_actions if hasattr(env, "n_actions") else env.action_space.n

    Net = DuelingQNet if dueling else QNet
    net = Net(obs_dim, n_actions)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()

    frames=[]
    if obs.ndim > 1: obs = obs.reshape(-1)
    for _ in range(steps):
        with torch.no_grad():
            a = net(torch.tensor(obs).unsqueeze(0)).argmax().item()
        obs, r, term, trunc, _ = env.step(a)
        frames.append(env.render())
        if obs.ndim > 1: obs = obs.reshape(-1)
        if term or trunc: break

    outp = Path(out); outp.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(outp, frames, fps=30)
    print("wrote", outp)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="submission\\snake_demo.mp4")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dueling", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    a=p.parse_args()
    record(a.ckpt, a.config, a.out, a.steps, a.dueling, a.seed)
