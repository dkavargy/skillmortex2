"""Fig. 2  Occupations and programming skills in two facing columns (redraw of the old network figure).

Ribbon width = the skill's relative share in that occupation: ads that mention the skill divided by the sum of
that count over the 13 programming skills, averaged over the 78 months. Each occupation's ribbons sum to 100%.
This measure is not affected by the daily export cap, unlike the Equivalence Index of the first submission.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
import smxfig as S
S.style()
sk, tt, prog = S.base_tables()
P = S.all_pairs()
M = pd.DataFrame({occ: {skl: P[(skl, occ)]["s"].mean() for skl in S.PROG_SKILLS} for occ in S.OCCUPATIONS})   # skill x occupation, %
M = M / M.sum() * 100
ads = tt[tt["month"].isin(set(S.MLIST))].groupby("occupation4d")["den"].sum()
order = list(M.sum(axis=1).sort_values(ascending=False).index)                 # skills, largest total first
M = M.loc[order]
COL = dict(zip(S.OCCUPATIONS, S.IDENT))

GL, GR = 34.0, 8.5                                                              # gaps between nodes, in share points
H = 400 + 3 * GL; assert abs(H - (400 + 12 * GR)) < 1e-9
XL0, XL1, XR0, XR1 = 0.205, 0.217, 0.525, 0.537                                 # node bars
fig = plt.figure(figsize=(7.2, 5.5))
ax = fig.add_axes([0, 0, 1, 0.93]); ax.set_xlim(0, 1); ax.set_ylim(H, -2); ax.axis("off")

ytopL = {occ: j * (100 + GL) for j, occ in enumerate(S.OCCUPATIONS)}
ytopR, y = {}, 0.0
for skl in order:
    ytopR[skl] = y; y += M.loc[skl].sum() + GR

def ribbon(x0, a0, a1, x1, b0, b1, color):
    xm = (x0 + x1) / 2
    verts = [(x0, a0), (xm, a0), (xm, b0), (x1, b0), (x1, b1), (xm, b1), (xm, a1), (x0, a1), (x0, a0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=0.50, zorder=2))

curR = dict(ytopR)
for occ in S.OCCUPATIONS:
    yl = ytopL[occ]
    ax.add_patch(Rectangle((XL0, yl), XL1 - XL0, 100, facecolor=COL[occ], edgecolor="none", zorder=3))
    for skl in order:
        v = M.loc[skl, occ]
        if v > 0:
            ribbon(XL1, yl, yl + v, XR0, curR[skl], curR[skl] + v, COL[occ])
        yl += v; curR[skl] += v
    ax.text(XL0 - 0.012, ytopL[occ] + 42, S.OCC_SHORT[occ], ha="right", va="center", fontsize=8, fontweight="semibold", color=S.INK)
    ax.text(XL0 - 0.012, ytopL[occ] + 58, f"{ads[occ] / 1e6:.2f} million ads", ha="right", va="center", fontsize=6.6, color=S.INK2)

XT = [0.808, 0.866, 0.924, 0.982]                                               # right edge of each table column
for skl in order:
    h = M.loc[skl].sum(); yc = ytopR[skl] + h / 2
    ax.add_patch(Rectangle((XR0, ytopR[skl]), XR1 - XR0, h, facecolor=S.INK2, edgecolor="none", zorder=3))
    ax.text(XR1 + 0.010, yc, S.SKILL_LABEL[skl], ha="left", va="center", fontsize=7.4, color=S.INK)
    top = M.loc[skl].idxmax()
    for x, occ in zip(XT, S.OCCUPATIONS):
        v = M.loc[skl, occ]
        if P[(skl, occ)]["observed"]:
            ax.text(x, yc, f"{v:.1f}", ha="right", va="center", fontsize=7, color=S.INK if occ == top else S.INK2,
                    fontweight="bold" if occ == top else "normal")
        else:
            ax.text(x, yc, "–", ha="right", va="center", fontsize=7, color=S.MUTED)
    ax.plot([XR1 + 0.008, XT[-1]], [ytopR[skl] + h + GR / 2] * 2, color=S.GRID, lw=0.4, zorder=1)

# column heads (figure coordinates, above the axes)
fy = lambda yy: 0.93 * (1 - (yy + 2) / (H + 2))
fig.text(XL0 - 0.012, 0.965, "Occupation", ha="right", va="bottom", fontsize=7.2, color=S.MUTED, fontweight="semibold")
fig.text(XR1 + 0.010, 0.965, "Programming skill", ha="left", va="bottom", fontsize=7.2, color=S.MUTED, fontweight="semibold")
fig.text(XT[-1], 0.990, "Relative share in the occupation, %", ha="right", va="bottom", fontsize=7.2, color=S.MUTED, fontweight="semibold")
for x, occ in zip(XT, S.OCCUPATIONS):
    fig.patches.append(Rectangle((x - 0.046, 0.950), 0.046, 0.007, transform=fig.transFigure, facecolor=COL[occ], edgecolor="none"))
    fig.text(x, 0.962, {"Software developers": "Software", "Systems analysts": "Systems", "Web and multimedia developers": "Web",
                        "Applications programmers": "Applic."}[occ], ha="right", va="bottom", fontsize=6.6, color=S.INK2)
S.save(fig, "fig2_occupation_skill")
print(M.round(1).to_string()); print(M.sum().round(1).to_dict())
