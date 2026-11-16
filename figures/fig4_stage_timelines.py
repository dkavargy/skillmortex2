"""Fig. 4  Stage timelines for every skill-occupation pair (relative share, one rule)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import smxfig as S
S.style()
P = S.all_pairs()
order = S.skill_order()[::-1]            # growing skills on top, declining at the bottom
nr, nc = len(order), len(S.OCCUPATIONS)

fig = plt.figure(figsize=(7.2, 7.0))
gs = GridSpec(nr, nc, figure=fig, left=0.215, right=0.985, top=0.918, bottom=0.048, hspace=0.30, wspace=0.10)
xend = S.MDATE[-1] + pd.offsets.MonthBegin(1)

for i, skl in enumerate(order):
    for j, occ in enumerate(S.OCCUPATIONS):
        ax = fig.add_subplot(gs[i, j]); d = P[(skl, occ)]
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_yticks([]); ax.set_xlim(S.MDATE[0], xend)
        if i == nr - 1:
            ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in (2019, 2021, 2023, 2025)])
            ax.set_xticklabels(["'19", "'21", "'23", "'25"], fontsize=6.3); ax.tick_params(axis="x", length=2, pad=1.5)
        else:
            ax.set_xticks([])
        if i == 0:
            ax.annotate(S.OCC_SHORT[occ], xy=(0, 1), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                        fontsize=7.6, fontweight="semibold", va="bottom", ha="left")
        if j == 0:
            ax.annotate(S.SKILL_LABEL[skl], xy=(0, 0.5), xycoords="axes fraction", xytext=(-8, 0), textcoords="offset points",
                        fontsize=7.4, va="center", ha="right")
        if not d["observed"]:
            ax.axvspan(S.MDATE[0], xend, facecolor="white", edgecolor=S.AXIS, hatch="////", lw=0)
            ax.annotate("not observed", xy=(0.5, 0.5), xycoords="axes fraction", fontsize=6.2, color=S.MUTED, ha="center", va="center",
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
            ax.set_ylim(0, 1); continue
        s, L, st = d["s"], d["L"], d["stages"]
        top = max(np.nanmax(L), np.nanpercentile(s, 97)) * 1.18
        ax.set_ylim(-0.20 * top, top)
        for a, b, c in S.runs(st):                       # stage strip along the bottom of the cell
            x0, x1 = S.MDATE[a], (S.MDATE[b] if b < len(st) else xend)
            if c == S.NYO:
                continue
            ax.fill_between([x0, x1], -0.20 * top, -0.05 * top, color=S.STAGE_COL[S.STAGES[c]], lw=0, zorder=2)
            ax.plot([x0, x0], [-0.20 * top, -0.05 * top], color="white", lw=0.8, zorder=3)
        ax.plot(S.MDATE, np.clip(s, None, top), color=S.MUTED, lw=0.45, alpha=0.8, zorder=3)
        ax.plot(S.MDATE, L, color=S.INK, lw=1.15, zorder=4)
        low = np.nanmin(L[-24:]) / top < 0.50             # line runs low on the right: put the number above it
        ax.annotate(f"{L[-1]:.1f}%", xy=(S.MDATE[-1], (L[-1] if low else 0)), xytext=(0, 4 if low else 2.5), textcoords="offset points",
                    fontsize=5.8, color=S.INK2, ha="right", va="bottom")

S.stage_legend(fig, 0.215, 0.970, stages=("Growth", "Maturity", "Decline", "Revival"), extra="Not observed", dx=0.103, fontsize=6.8)
fig.text(0.985, 0.976, "Number: relative share now", fontsize=6.2, color=S.MUTED, ha="right", va="center")
S.save(fig, "fig4_stage_timelines")
print(order)
