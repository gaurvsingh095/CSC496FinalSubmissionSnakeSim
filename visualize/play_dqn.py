import argparse, time, numpy as np, torch, pygame
from envs.snake_env import SnakeEnv
from agents.dqn import QNet

def load_policy(ckpt, in_ch, H, W, nA, device="cpu"):
    q = QNet(in_ch, H, W, nA).to(device)
    q.load_state_dict(torch.load(ckpt, map_location=device))
    q.eval()
    return q

def main(args):
    pygame.init()
    scale = args.scale
    env = SnakeEnv(size=args.size, feature_state=False, frame_stack=args.frames)
    o,_ = env.reset()
    in_ch, H, W = o.shape[0], o.shape[1], o.shape[2]
    nA = env.action_space.n
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    q = load_policy(args.ckpt, in_ch, H, W, nA, device=device)

    screen = pygame.display.set_mode((env.size*scale, env.size*scale))
    pygame.display.set_caption("Snake RL — DQN Play")
    clock = pygame.time.Clock()

    def act(obs):
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0).to(device)
            return int(q(t).argmax(1).item())

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        a = act(o)
        o, r, done, tr, _ = env.step(a)
        frame = env.render_array(scale)
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1,0,2)))
        screen.blit(surf, (0,0))
        pygame.display.flip()

        clock.tick(args.fps)
        if done:
            pygame.time.wait(300)
            o,_ = env.reset()

    pygame.quit()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to checkpoints/dqn_latest.pt")
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--scale", type=int, default=24)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()
    main(args)
