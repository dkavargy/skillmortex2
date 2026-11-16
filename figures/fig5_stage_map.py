"""Fig. 5  Stage of every pair with its uncertainty (joint threshold draws) and its robustness (countries, sources)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import smxfig as S
S.style()
P = S.all_pairs(); prob = S.stage_probabilities()
rob = S.tab("tab10_stage_robustness.csv").set_index(["skill", "occupation"])
VARS = [("United Kingdom", "UK"), ("DEUTSCHLAND", "DE"), ("France", "FR"), ("Other", "Other\ncountries"),
        ("Fixed mix", "Fixed\ncountry mix"), ("src:stable", "Stable\nsources"), ("src:unstable", "Unstable\nsources")]
order = S.skill_order()[::-1]
nr, nc = len(order), len(S.OCCUPATIONS)
SEG = ["Growth", "Revival", "Maturity", "Decline", "Death"]

fig = plt.figure(figsize=(7.2, 6.6))
ax = fig.add_axes([0.215, 0.03, 0.77, 0.815]); ax.set_xlim(0, nc); ax.set_ylim(nr, 0); ax.axis("off")
cw, ch = 0.90, 0.80                       # drawn width / height of a cell, in cell units
grades = {}
for i, skl in enumerate(order):
    ax.text(-0.06, i + 0.42, S.SKILL_LABEL[skl], ha="right", va="center", fontsize=7.4)
    if i:
        ax.plot([-1.45, nc - 0.06], [i - 0.07, i - 0.07], color=S.GRID, lw=0.5, clip_on=False)
    for j, occ in enumerate(S.OCCUPATIONS):
        x0, y0 = j, i
        if i == 0:
            ax.text(x0, -0.42, S.OCC_SHORT[occ], ha="left", va="bottom", fontsize=7.6, fontweight="semibold")
        d = P[(skl, occ)]
        if not d["observed"]:
            ax.add_patch(Rectangle((x0, y0 + 0.06), cw, 0.70, facecolor="white", edgecolor=S.AXIS, hatch="////", lw=0))
            ax.text(x0 + cw / 2, y0 + 0.41, "not observed", ha="center", va="center", fontsize=6.2, color=S.MUTED,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
            continue
        pr = prob.loc[(skl, occ)]; modal = pr.idxmax(); pm = pr.max()
        # variants
        vs = [rob.loc[(skl, occ), v] for v, _ in VARS]
        ok = [v for v in vs if v != "too few ads" and isinstance(v, str)]
        down = {"Decline", "Death"}                      # a re-run that says Death agrees with Decline
        same = sum((v == d["stage"]) or (v in down and d["stage"] in down) for v in ok)
        grade = "consistent" if (pm >= 0.85 and len(ok) >= 3 and same >= len(ok) - 1) else ("pooled" if pm >= 0.85 else "uncertain")
        grades[(skl, occ)] = (d["stage"], modal, round(pm, 2), same, len(ok), grade)
        # label
        ax.text(x0, y0 + 0.20, d["stage"], ha="left", va="center", fontsize=7.2,
                fontweight="bold" if grade == "consistent" else "normal", color=S.INK if grade != "uncertain" else S.INK2)
        ax.text(x0 + cw, y0 + 0.20, f"{pr[d['stage']]:.0%}", ha="right", va="center", fontsize=6.6, color=S.INK2)
        # probability bar
        xb, gap = x0, 0.010
        for sname in SEG:
            w = pr.get(sname, 0.0) * cw
            if w <= 0:
                continue
            ax.add_patch(Rectangle((xb + (gap / 2 if xb > x0 else 0), y0 + 0.34), max(w - gap / 2, 0.004), 0.20,
                                   facecolor=S.STAGE_COL[sname], edgecolor="none"))
            xb += w
        # variant squares
        sq, sgap = 0.085, 0.034
        for k, v in enumerate(vs):
            xs = x0 + k * (sq + sgap)
            if v == "too few ads" or not isinstance(v, str):
                ax.add_patch(Rectangle((xs, y0 + 0.62), sq, 0.115, facecolor="white", edgecolor=S.AXIS, lw=0.5))
            else:
                ax.add_patch(Rectangle((xs, y0 + 0.62), sq, 0.115, facecolor=S.STAGE_COL[v], edgecolor="none"))

# ---- key (top) -----------------------------------------------------------------------------------------
S.stage_legend(fig, 0.215, 0.979, stages=("Growth", "Maturity", "Decline", "Death"), extra="Not observed", dx=0.092, fontsize=6.8)
kx = fig.add_axes([0.215, 0.906, 0.77, 0.056]); kx.set_xlim(0, nc); kx.set_ylim(1, 0); kx.axis("off")
kx.text(0, 0.14, "Maturity", fontsize=7.2, fontweight="bold", va="center"); kx.text(cw, 0.14, "86%", fontsize=6.6, color=S.INK2, va="center", ha="right")
kx.add_patch(Rectangle((0, 0.36), 0.86 * cw - 0.005, 0.26, facecolor=S.STAGE_COL["Maturity"], edgecolor="none"))
kx.add_patch(Rectangle((0.86 * cw + 0.005, 0.36), 0.14 * cw - 0.005, 0.26, facecolor=S.STAGE_COL["Decline"], edgecolor="none"))
for k in range(7):
    fc = [S.STAGE_COL["Decline"], S.STAGE_COL["Maturity"], "white"][0 if k == 0 else (2 if k == 2 else 1)]
    kx.add_patch(Rectangle((k * (0.085 + 0.034), 0.78), 0.085, 0.16, facecolor=fc, edgecolor=S.AXIS if k == 2 else "none", lw=0.5))
tx = cw + 0.08
kx.text(tx, 0.14, "Stage and its share of the 2,000 threshold draws. Bold: also holds in the re-runs. Grey: under 85% of draws",
        fontsize=6.1, color=S.INK2, va="center")
kx.text(tx, 0.49, "Share of draws giving each stage", fontsize=6.1, color=S.INK2, va="center")
kx.text(tx, 0.86, "Re-runs: UK, DE, FR, other, fixed country mix, stable and unstable sources; hollow: too few ads", fontsize=6.1, color=S.INK2, va="center")
S.save(fig, "fig5_stage_map")
g = pd.DataFrame(grades, index=["stage", "modal", "p", "same", "n", "grade"]).T
g.index.names = ["skill", "occupation"]; g.to_csv(__import__("os").path.join(S.OUT, "evidence_grades_by_pair.csv"))
print(g["grade"].value_counts().to_dict()); print(g[g.grade == "consistent"].groupby("stage").size().to_dict())
