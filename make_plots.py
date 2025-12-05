import argparse, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--csv", default="runs/episodes.csv")
p.add_argument("--outdir", default="submission")
p.add_argument("--last", type=int, default=200)     # how many recent episodes to plot
p.add_argument("--smooth", type=float, default=0.9) # EMA smoothing [0..1)
a = p.parse_args()

df = pd.read_csv(a.csv)
df = df.sort_values(["config","seed","episode"])
# take the most recent block for your current config/seed
g = df.groupby(["config","seed"])
tail = g.tail(a.last)

def ema(s, alpha):
    out = []
    m = None
    for v in s:
        m = v if m is None else alpha*m + (1-alpha)*v
        out.append(m)
    return out

outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

for (cfg, sd), block in tail.groupby(["config","seed"]):
    block = block.sort_values("episode")
    # Return plot
    plt.figure()
    plt.plot(block["episode"], block["return"], label="return (raw)")
    plt.plot(block["episode"], ema(block["return"], a.smooth), label=f"EMA {a.smooth}")
    plt.xlabel("episode"); plt.ylabel("return"); plt.title(f"{cfg} | seed {sd}")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"plot_return_seed{sd}.png"); plt.close()

    # Length plot
    plt.figure()
    plt.plot(block["episode"], block["length"], label="length (raw)")
    plt.plot(block["episode"], ema(block["length"], a.smooth), label=f"EMA {a.smooth}")
    plt.xlabel("episode"); plt.ylabel("steps"); plt.title(f"{cfg} | seed {sd}")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"plot_length_seed{sd}.png"); plt.close()

print("Saved plots to", outdir.resolve())
