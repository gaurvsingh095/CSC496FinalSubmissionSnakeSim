# video_recorder.py
import imageio.v2 as imageio
import torch, yaml
from pathlib import Path
from envs.snake_env import SnakeEnv
from agents.nets import QNet, DuelingQNet

def _resolve_path(p: str) -> Path:
    pth = Path(p)
    if pth.is_file():
        return pth
    alt = (Path(__file__).parent / p).resolve()
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"Config not found: {p} (tried {pth} and {alt})")

def record(ckpt, config_path, out="demo.mp4", steps=1000, dueling=False, seed=123):
    cfg_file = _resolve_path(config_path)
    ckpt_file = Path(ckpt)
    if not ckpt_file.is_file():
        alt_ckpt = (Path(__file__).parent / ckpt).resolve()
        if alt_ckpt.is_file():
            ckpt_file = alt_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    cfg = yaml.safe_load(cfg_file.read_text())
    env = SnakeEnv(**cfg["env"], seed=seed, render_mode="rgb_array")
    Net = DuelingQNet if dueling else QNet
    net = Net(env.obs_dim, env.n_actions)
    net.load_state_dict(torch.load(str(ckpt_file), map_location="cpu"))
    net.eval()

    frames=[]
    obs,_ = env.reset(seed=seed)
    if obs.ndim > 1:
        obs = obs.reshape(-1)
    for _ in range(steps):
        with torch.no_grad():
            a = net(torch.tensor(obs).unsqueeze(0)).argmax().item()
        obs, r, term, trunc, _ = env.step(a)
        frame = env.render()  # rgb_array
        frames.append(frame)
        if obs.ndim > 1:
            obs = obs.reshape(-1)
        if term or trunc:
            break

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=30)
    print("wrote", out_path)

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
