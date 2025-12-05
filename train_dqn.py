import argparse, os, csv, yaml
import numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from envs.snake_env import SnakeEnv
from agents.nets import QNet, DuelingQNet        # MLP nets: QNet(obs_dim,n_actions)
from agents.replay import ReplayBuffer, PERBuffer

# ---------------- utils ----------------
def linear_eps(step, hi, lo, decay_steps):
    t = min(1.0, step / float(decay_steps))
    return hi * (1 - t) + lo * t

def build_net(obs_dim, n_actions, dueling=False):
    return (DuelingQNet if dueling else QNet)(obs_dim, n_actions)

def build_buffer(cap, per=False, alpha=0.6, beta0=0.4):
    return PERBuffer(cap, alpha, beta0) if per else ReplayBuffer(cap)

@torch.no_grad()
def ddqn_target(online, target, next_states, rewards, dones, gamma):
    # Double-DQN: action from online, value from target
    next_a = online(next_states).argmax(dim=1, keepdim=True)
    next_q = target(next_states).gather(1, next_a).squeeze(1)
    return rewards + gamma * (1.0 - dones) * next_q

def maybe_resize_env(env, step, curriculum):
    # curriculum = [{"until": 80_000, "size": 8}, {"until": 160_000, "size": 10}, {"until": 9_999_999, "size": 12}]
    if not curriculum:
        return
    for stage in curriculum:
        if step < stage["until"] and getattr(env, "size", None) != stage["size"]:
            if hasattr(env, "resize"):
                env.resize(stage["size"])
            break

# --------------- main ------------------
def main(args):
    # ---- load config ----
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    log_cfg = cfg.get("log", {})

    # Use FEATURE state (vector) so we can use the MLP QNet
    size           = args.size or env_cfg.get("board_size", 8)
    feature_state  = env_cfg.get("feature_state", True)  # <- TRUE for MLP
    frames         = args.frames or env_cfg.get("frame_stack", 1)
    step_penalty   = env_cfg.get("step_penalty", -0.003)

    total_steps    = args.steps or train_cfg.get("total_steps", 300_000)
    batch_size     = train_cfg.get("batch_size", 128)
    buffer_size    = train_cfg.get("buffer_size", 100_000)
    warmup_steps   = train_cfg.get("warmup_steps", 10_000)
    gamma          = train_cfg.get("gamma", 0.99)
    lr             = train_cfg.get("lr", 5e-4)
    target_sync    = train_cfg.get("target_sync", 2000)
    eps_start      = train_cfg.get("eps_start", 1.0)
    eps_end        = train_cfg.get("eps_end", 0.05)
    eps_decay      = train_cfg.get("eps_decay_steps", 150_000)
    dueling        = train_cfg.get("dueling", True)
    use_per        = train_cfg.get("per", False)
    alpha          = train_cfg.get("alpha", 0.6)
    beta0          = train_cfg.get("beta0", 0.4)
    curriculum     = train_cfg.get("curriculum", None)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- env & nets ----
    env = SnakeEnv(size=size, feature_state=feature_state, frame_stack=frames, step_penalty=step_penalty)
    obs, _ = env.reset(seed=args.seed)
    if obs.ndim == 1:
        obs_dim = obs.shape[0]
    else:
        # flatten just in case (still works with feature_state=True)
        obs_dim = int(np.prod(obs.shape))
        obs = obs.reshape(-1)
    nA = env.action_space.n

    # build networks
    q_online = build_net(obs_dim, nA, dueling=dueling).to(device)
    q_target = build_net(obs_dim, nA, dueling=dueling).to(device)
    q_target.load_state_dict(q_online.state_dict()); q_target.eval()
    opt = optim.Adam(q_online.parameters(), lr=lr)

    # replay
    buf = build_buffer(buffer_size, per=use_per, alpha=alpha, beta0=beta0)

    # logging
    writer = SummaryWriter(log_dir="runs") if log_cfg.get("tensorboard", True) else None
    csv_path = args.log_csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["config","seed","episode","return","length","epsilon","loss"])

    os.makedirs("checkpoints", exist_ok=True)

    rng = np.random.default_rng(args.seed)
    ep_ret, ep_len, ep = 0.0, 0, 0
    loss_val = 0.0
    o = obs.copy()

    for step in trange(total_steps, desc="Training"):
        # optional curriculum
        maybe_resize_env(env, step, curriculum)

        # epsilon-greedy
        eps = linear_eps(step, eps_start, eps_end, eps_decay)
        if rng.random() < eps:
            a = rng.integers(nA)
        else:
            with torch.no_grad():
                x = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                a = int(q_online(x).argmax(1).item())

        # step
        o2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        buf.add((o, a, r, o2, float(done)))
        o = o2.reshape(-1) if o2.ndim > 1 else o2
        ep_ret += r; ep_len += 1

        # learn
        if len(buf) >= warmup_steps:
            if use_per:
                batch, idx, w = buf.sample(batch_size)
                w = torch.tensor(w, dtype=torch.float32, device=device)
            else:
                batch, idx = buf.sample(batch_size)
                w = None

            s, a_b, r_b, s2, d = map(list, zip(*batch))
            s   = torch.tensor(np.array(s),  dtype=torch.float32, device=device)
            a_b = torch.tensor(a_b,          dtype=torch.long,   device=device).view(-1,1)
            r_b = torch.tensor(r_b,          dtype=torch.float32,device=device)
            s2  = torch.tensor(np.array(s2), dtype=torch.float32,device=device)
            d   = torch.tensor(d,            dtype=torch.float32,device=device)

            q_sa = q_online(s).gather(1, a_b).squeeze(1)
            target = ddqn_target(q_online, q_target, s2, r_b, d, gamma)

            if w is None:
                loss = nn.SmoothL1Loss()(q_sa, target)
            else:
                td = torch.abs(target - q_sa).detach().cpu().numpy()
                buf.update_priorities(idx, td)
                per_loss = nn.SmoothL1Loss(reduction="none")(q_sa, target)
                loss = (w * per_loss).mean()

            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q_online.parameters(), 10.0); opt.step()
            loss_val = float(loss.item())
            if (step + 1) % target_sync == 0:
                q_target.load_state_dict(q_online.state_dict())
                torch.save(q_online.state_dict(), "checkpoints/dqn_latest.pt")

            if writer:
                writer.add_scalar("loss/td_loss", loss_val, step)

        # episode end
        if done:
            if writer:
                writer.add_scalar("charts/episode_return", ep_ret, ep)
                writer.add_scalar("charts/episode_length", ep_len, ep)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([args.config, args.seed, ep, ep_ret, ep_len, eps, loss_val])
            o, _ = env.reset(seed=int(rng.integers(1<<31)))
            o = o.reshape(-1) if o.ndim > 1 else o
            ep_ret, ep_len, ep = 0.0, 0, ep + 1

    torch.save(q_online.state_dict(), "checkpoints/dqn_final.pt")
    if writer: writer.flush(); writer.close()
    print("Training complete. Checkpoints in ./checkpoints")

# --------------- cli -------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dqn_small.yaml")
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--log_csv", type=str, default="runs/episodes.csv")
    args = p.parse_args()
    main(args)
