import numpy as np

def greedy_action(env):
    """Heuristic: move toward apple if that move isn't an immediate collision."""
    head = env.snake[0]
    fr, fc = env.apple
    best = None
    best_score = 1e9
    for a, (dr, dc) in {0:(-1,0),1:(0,1),2:(1,0),3:(0,-1)}.items():
        if (a + 2) % 4 == env.direction:
            continue
        nh = (head[0]+dr, head[1]+dc)
        if env._collides(nh):
            continue
        score = abs(fr - nh[0]) + abs(fc - nh[1])
        if score < best_score:
            best_score = score
            best = a
    if best is None:
        best = env.direction
    return best
