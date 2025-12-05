import argparse, numpy as np
from envs.snake_env import SnakeEnv
from baselines.tabular_q import TabularQ
from baselines.greedy import greedy_action
from tqdm import trange

def run(args):
    env = SnakeEnv(size=args.size, feature_state=True, frame_stack=1)
    agent = TabularQ()
    returns = []

    for ep in trange(args.episodes, desc="Tabular Q"):
        o,_ = env.reset()
        done, ret = False, 0.0
        while not done:
            a = agent.act(o)
            o2, r, d, tr, _ = env.step(a)
            agent.step(o, a, r, o2, d)
            o = o2; done = d; ret += r
        returns.append(ret)
        if (ep+1) % 500 == 0:
            avg = float(np.mean(returns[-500:]))
            print(f"Episode {ep+1}: avg_return(500)={avg:.3f}, eps={agent.eps:.3f}")

    # quick greedy comparison
    eval_ret = 0.0; E = 100
    for _ in range(E):
        o,_ = env.reset()
        done = False
        while not done:
            a = greedy_action(env)
            o, r, d, tr, _ = env.step(a)
            eval_ret += r
    print(f"[Greedy heuristic] mean return over {E} episodes = {eval_ret/E:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--episodes", type=int, default=20000)
    args = p.parse_args()
    run(args)
