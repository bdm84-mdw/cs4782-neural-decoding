"""Generate the headline R^2 comparison figure from results_table.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
df = pd.read_csv(HERE / "results_table.csv")
df = df.sort_values("adapted_day2_r2", ascending=True).reset_index(drop=True)

colors = ["#7f7f7f" if "POYO (event" not in m else "#d62728" for m in df["model"]]

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.barh(df["model"], df["adapted_day2_r2"], color=colors, edgecolor="black", linewidth=0.6)
for bar, v in zip(bars, df["adapted_day2_r2"]):
    ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=10)
ax.set_xlabel(r"Held-out Day-2 $R^2$ after adaptation")
ax.set_title("POYO vs. binned baselines (same animal, new day)")
ax.set_xlim(0, 0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "r2_comparison.png", dpi=200)
print(f"Wrote {HERE / 'r2_comparison.png'}")
