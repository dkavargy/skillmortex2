"""Fig. 3  How a share series becomes a stage timeline: three pairs, one rule."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import smxfig as S
S.style()
P = S.all_pairs()
halo = [pe.withStroke(linewidth=2.2, foreground="white")]

EX = [("css", "Web and multimedia developers"), ("ict system programming", "Systems analysts"), ("abap", "Software developers")]
fig = plt.figure(figsize=(7.2, 3.0))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.085], hspace=0.07, wspace=0.26, top=0.765, bottom=0.12, left=0.07, right=0.985)

for j, key in enumerate(EX):
    d = P[key]; s, L, Pk, st = d["s"], d["L"], d["P"], d["stages"]
    ax = fig.add_subplot(gs[0, j]); sx = fig.add_subplot(gs[1, j], sharex=ax)
    th = S.BASE["theta"] * Pk
    ax.fill_between(S.MDATE, th, Pk, color=S.STAGE_COL["Maturity"], alpha=0.28, lw=0, zorder=1)
    ax.plot(S.MDATE, s, color=S.MUTED, lw=0.6, zorder=2)
    ax.plot(S.MDATE, Pk, color=S.INK2, lw=0.7, drawstyle="steps-post", zorder=3)
    ax.plot(S.MDATE, L, color=S.INK, lw=1.9, zorder=4)
    ymax = np.nanmax(s) * 1.12
    ax.set_ylim(0, ymax); S.year_axis(ax, every=1, start=2019)
    ax.tick_params(axis="x", labelbottom=False, length=0); ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_visible(False); S.hgrid(ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    if j == 0:
        ax.set_ylabel("Relative share")
    path = " → ".join(S.STAGES[c] for a, b, c in S.runs(st) if c != S.NYO)
    ax.annotate(S.SKILL_LABEL[key[0]], xy=(0, 1), xycoords="axes fraction", xytext=(11, 26),
                textcoords="offset points", fontsize=8, fontweight="semibold", va="bottom")
    ax.annotate("abc"[j], xy=(0, 1), xycoords="axes fraction", xytext=(0, 25.4), textcoords="offset points", fontsize=9, fontweight="bold", va="bottom")
    ax.annotate(S.OCC_SHORT[key[1]], xy=(0, 1), xycoords="axes fraction", xytext=(0, 14), textcoords="offset points",
                fontsize=6.6, color=S.INK2, va="bottom")
    ax.annotate(path, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", fontsize=6.6, color=S.INK, va="bottom")
    # stage strip
    for a, b, c in S.runs(st):
        x0, x1 = S.MDATE[a], (S.MDATE[b] if b < len(st) else S.MDATE[-1] + pd.offsets.MonthBegin(1))
        if c == S.NYO:
            sx.axvspan(x0, x1, facecolor="white", edgecolor=S.AXIS, hatch="////", lw=0)
        else:
            sx.axvspan(x0, x1, facecolor=S.STAGE_COL[S.STAGES[c]], lw=0)
            sx.axvline(x0, color="white", lw=1.2)
    sx.set_ylim(0, 1); sx.set_yticks([]); sx.set_xlim(S.MDATE[0], S.MDATE[-1] + pd.offsets.MonthBegin(1))
    for sp in sx.spines.values():
        sp.set_visible(False)
    sx.tick_params(axis="x", length=0, pad=3)
    first = int(np.argmax(st != S.NYO))
    sx.annotate("warm-up", xy=(S.MDATE[first // 2], 0.5), fontsize=5.8, color=S.MUTED, ha="center", va="center", path_effects=halo)

    if j == 0:      # name the ingredients once
        lead = dict(arrowstyle="-", color=S.MUTED, lw=0.5, shrinkA=1.5, shrinkB=1.5)
        T = lambda m: pd.Timestamp(m + "-01")
        i1 = 5;  ax.annotate("Monthly share", xy=(S.MDATE[i1], s[i1]), xytext=(T("2018-12"), 4.6), fontsize=6.5, color=S.INK2,
                             ha="left", va="top", arrowprops=lead)
        i4 = 34; ax.annotate("80% of the peak", xy=(S.MDATE[i4], th[i4]), xytext=(S.MDATE[i4], 3.3), fontsize=6.5, color=S.INK2,
                             ha="center", va="top", arrowprops=lead)
        i2 = 56; ax.annotate("Level: 12-month median", xy=(S.MDATE[i2], L[i2]), xytext=(S.MDATE[i2], 2.0), fontsize=6.5, color=S.INK,
                             ha="center", va="top", arrowprops=lead)
        i3 = 50; ax.annotate("Running peak", xy=(S.MDATE[i3], Pk[i3]), xytext=(0, 9), textcoords="offset points", fontsize=6.5, color=S.INK2,
                             ha="center", arrowprops=lead)
    if j == 2:
        ip = int(np.argmax(s)); ax.annotate("Three-month burst:\nno new peak", xy=(S.MDATE[ip], s[ip]), xytext=(-62, -4), textcoords="offset points",
                                           fontsize=6.5, color=S.INK2, va="top", linespacing=1.25, path_effects=halo)

S.stage_legend(fig, 0.07, 0.955, stages=("Growth", "Maturity", "Decline", "Revival", "Death"), dx=0.092)
fig.text(0.985, 0.961, "A stage changes only after 3 consecutive months",
         fontsize=6.4, color=S.MUTED, ha="right", va="center")
S.save(fig, "fig3_stage_rule")
for key in EX:
    d = P[key]; print(key, [(str(S.MONTHS[a]), S.STAGES[c]) for a, b, c in S.runs(d["stages"])])
