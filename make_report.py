
from pathlib import Path
import pandas as pd

csv_path = Path("runs/episodes.csv")
if not csv_path.exists():
    raise SystemExit(f"No CSV found at {csv_path}")

df = pd.read_csv(csv_path)
# Use last K episodes per (config, seed)
K = 50
df['rank'] = df.groupby(['config','seed'])['episode'].rank(method='first', ascending=False)
tail = df[df['rank'] <= K]

agg = tail.groupby(['config','seed']).agg(
    episodes=('episode','max'),
    avg_return=('return','mean'),
    survival=('length','median'),
    return_std=('return','std'),
).reset_index()

out = Path("runs/summary.csv")
agg.to_csv(out, index=False)
print(agg)
print("Wrote", out.resolve())
