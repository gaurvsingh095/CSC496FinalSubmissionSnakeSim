import argparse, numpy as np, imageio.v2 as imageio, torch
from envs.snake_env import SnakeEnv
from agents.dqn import QNet

def main(args):
    env = SnakeEnv(size=args.size, feature_state=False, frame_stack=args.frames)
    o,_ = env.reset()
    in_ch, H, W = o.shape[0], o.shape[1], o.shape[2]
    nA = env.action_space.n

    q = QNet(in_ch, H, W, nA)
    q.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    q.eval()

    def act(obs):
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0).float()
            return int(q(t).argmax(1).item())

    frames = []
    for ep in range(args.episodes):
        o,_ = env.reset()
        done = False
        t = 0
        while not done and t < args.max_steps:
            a = act(o)
            o, r, done, tr, _ = env.step(a)
            frames.append(env.render_array(scale=args.scale))
            t += 1

    imageio.mimsave(args.out, frames, duration=1.0/args.fps)
    print(f"Saved GIF to {args.out} with {len(frames)} frames.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--scale", type=int, default=16)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", default="snake_play.gif")
    args = p.parse_args()
    main(args)
