"""Fig. 7  November 2022 against placebo dates: the same before-after comparison at every month."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import smxfig as S
S.style()
bd = S.tab("tab8_chatgpt_placebo_by_date.csv").rename(columns={"Unnamed: 0": "month"})
bp = S.tab("tab8_chatgpt_placebo_by_pair.csv").set_index(["skill", "occupation"])
bd["date"] = pd.to_datetime(bd["month"] + "-01")
ev = bd[bd["kind"] == "EVENT"].iloc[0]; pl = bd[bd["kind"] == "placebo"]
order = S.skill_order()[::-1]

fig = plt.figure(figsize=(7.2, 6.0))
gt = GridSpec(1, 2, figure=fig, left=0.075, right=0.985, top=0.925, bottom=0.735, wspace=0.22)
gbm = GridSpec(1, 4, figure=fig, left=0.215, right=0.985, top=0.585, bottom=0.085, wspace=0.07)

for k, (col, ttl, ylab, letter) in enumerate([
        ("mean_abs_change_%", "Size of the change at each date", "Mean absolute change (%)", "a"),
        ("pairs_up_5%", "Pairs the old rule would flag", "Pairs rising 5% or more", "b")]):
    ax = fig.add_subplot(gt[0, k])
    lo, hi = pl[col].quantile(0.05), pl[col].quantile(0.95)
    ax.axhspan(lo, hi, color=S.STAGE_COL["Maturity"], alpha=0.35, lw=0, zorder=1)
    ax.plot(bd["date"], bd[col], color=S.MUTED, lw=1.0, zorder=2)
    ax.scatter(pl["date"], pl[col], s=7, color=S.INK2, lw=0, zorder=3)
    ax.scatter([ev["date"]], [ev[col]], s=38, color=S.ACCENT, edgecolor="white", lw=1.0, zorder=5)
    ax.axvline(ev["date"], color=S.ACCENT, lw=0.7, zorder=1)
    nhi = int((pl[col] >= ev[col]).sum())
    ax.annotate(f"Nov 2022: {ev[col]:.1f}" if k == 0 else f"Nov 2022: {int(ev[col])} of 48", xy=(ev["date"], ev[col]),
                xytext=(10, 14) if k == 0 else (9, 27), textcoords="offset points", fontsize=6.8, color=S.INK, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.5, shrinkB=3))
    ax.annotate(f"{nhi} of {len(pl)} placebo dates are at least as high", xy=(0, 1), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", fontsize=6.4, color=S.INK2, ha="left", va="bottom")
    ax.set_ylim(0, max(bd[col].max() * (1.22 if k == 0 else 1.36), hi * 1.3)); ax.set_ylabel(ylab)
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2020, 2025)]); ax.set_xticklabels([str(y) for y in range(2020, 2025)])
    ax.set_xlim(bd["date"].min() - pd.offsets.MonthBegin(2), bd["date"].max() + pd.offsets.MonthBegin(2))
    S.hgrid(ax); ax.tick_params(axis="y", length=0)
    S.panel(ax, letter, ttl, dy=17)
    if k == 0:
        ax.annotate("5th–95th percentile of placebo dates", xy=(pd.Timestamp("2019-05-01"), lo), xytext=(0, -3), textcoords="offset points",
                    fontsize=6.0, color=S.MUTED, ha="left", va="top")

# ---- c: every pair, change at the event against its own placebo range -----------------------------------
CL = 62
for j, occ in enumerate(S.OCCUPATIONS):
    ax = fig.add_subplot(gbm[0, j])
    ax.axvline(0, color=S.AXIS, lw=0.7, zorder=1)
    for i, skl in enumerate(order):
        if (skl, occ) not in bp.index:
            ax.annotate("not observed", xy=(0, i), fontsize=5.8, color=S.MUTED, ha="center", va="center"); continue
        r = bp.loc[(skl, occ)]
        a, b, v = r["placebo_p05_%"], r["placebo_p95_%"], r["change_at_event_%"]
        ax.plot([max(a, -CL), min(b, CL)], [i, i], color=S.STAGE_COL["Maturity"], lw=4.2, solid_capstyle="butt", zorder=2)
        for edge, sign in ((a, -1), (b, 1)):
            if abs(edge) > CL:
                ax.scatter([sign * (CL + 3.5)], [i], marker=">" if sign > 0 else "<", s=14, color=S.MUTED, lw=0, zorder=3, clip_on=False)
        unusual = r["share_of_placebo_dates_as_large"] < 0.10
        ax.scatter([np.clip(v, -CL, CL)], [i], s=26 if unusual else 16, color=S.ACCENT if unusual else S.INK, edgecolor="white", lw=0.8, zorder=4)
        if unusual:
            ax.annotate(f"{v:+.0f}%".replace("-", "\u2212"), xy=(np.clip(v, -CL, CL), i), xytext=(-5 if v < 0 else 5, 0),
                        textcoords="offset points", fontsize=6.0, color=S.INK2, ha="right" if v < 0 else "left", va="center",
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9), zorder=5)
    ax.set_ylim(len(order) - 0.45, -0.6); ax.set_xlim(-CL - 8, CL + 8)
    ax.set_xticks([-50, 0, 50]); ax.set_xticklabels(["−50", "0", "+50%"])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([S.SKILL_LABEL[s] for s in order] if j == 0 else [], fontsize=7.2)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_title(S.OCC_SHORT[occ], fontsize=7.6, pad=5)
    for i in range(len(order)):
        ax.axhline(i + 0.5, color=S.GRID, lw=0.4, zorder=0)
    if j == 0:
        ax.annotate("c", xy=(0, 1), xycoords="axes fraction", xytext=(-108, 27), textcoords="offset points", fontsize=9, fontweight="bold", va="bottom")
        ax.annotate("Change at November 2022 for each pair, against its own range at placebo dates", xy=(0, 1), xycoords="axes fraction",
                    xytext=(-97, 27.6), textcoords="offset points", fontsize=8, fontweight="semibold", va="bottom")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=S.STAGE_COL["Maturity"], lw=4.2, label="5th–95th percentile at placebo dates"),
                    Line2D([], [], marker="o", color="none", markerfacecolor=S.INK, markeredgecolor="white", markersize=5, label="Change at November 2022"),
                    Line2D([], [], marker="o", color="none", markerfacecolor=S.ACCENT, markeredgecolor="white", markersize=6,
                           label="Larger than at 90% of placebo dates")],
           loc="upper left", bbox_to_anchor=(0.215, 0.648), ncol=3, fontsize=6.4, columnspacing=1.6, handletextpad=0.5, borderaxespad=0)
fig.text(0.6, 0.012, "Change in relative share, six months after against six months before", fontsize=7.5, color=S.INK2, ha="center", va="bottom")
S.save(fig, "fig7_placebo")
print("event", ev[["mean_abs_change_%", "pairs_up_5%"]].to_dict(), "| unusual", int((bp["share_of_placebo_dates_as_large"] < 0.10).sum()))
