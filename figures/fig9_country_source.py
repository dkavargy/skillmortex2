"""Fig. 9  Who supplies the ads changes over time, and how far the stages survive it."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import smxfig as S
S.style()
mset = set(S.MLIST)
ttc = S.rd("totals_country"); ttc = ttc[ttc["month"].isin(mset)]
tts = S.rd("totals_stab");    tts = tts[tts["month"].isin(mset)]
NAME = {"United Kingdom": "United Kingdom", "DEUTSCHLAND": "Germany", "France": "France"}
tot = ttc.groupby("month")["n_post"].sum()
ncountry = ttc["country"].nunique()
grp = ttc.assign(g=ttc["country"].map(NAME).fillna("Other")).groupby(["month", "g"])["n_post"].sum().unstack().reindex(S.MLIST)
share = grp.div(tot.reindex(S.MLIST), axis=0) * 100
stab = tts.groupby(["month", "source_stability"])["n_post"].sum().unstack().reindex(S.MLIST)
stable = stab["stable"] / stab.sum(axis=1) * 100

fig = plt.figure(figsize=(7.2, 4.8))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.18], hspace=0.62, left=0.065, right=0.985, top=0.835, bottom=0.085)
top = gs[0].subgridspec(1, 6, wspace=0.14, width_ratios=[1, 1, 1, 1, 0.16, 1.0])
bot = gs[1].subgridspec(1, 1)

def ends(ax, y, fmt="{:.0f}%"):
    """Value labels at the first and last twelve-month means."""
    a, b = float(np.mean(y[:12])), float(np.mean(y[-12:]))
    ax.plot([S.MDATE[0], S.MDATE[11]], [a, a], color=S.ACCENT, lw=1.6, solid_capstyle="butt", zorder=4)
    ax.plot([S.MDATE[-12], S.MDATE[-1]], [b, b], color=S.ACCENT, lw=1.6, solid_capstyle="butt", zorder=4)
    return f"{a:.0f}% \u2192 {b:.0f}%"

cols = ["United Kingdom", "Germany", "France", "Other"]
for j, c in enumerate(cols):
    ax = fig.add_subplot(top[0, j])
    y = share[c].to_numpy()
    ax.plot(S.MDATE, y, color=S.INK, lw=1.0, zorder=3)
    ax.fill_between(S.MDATE, 0, y, color=S.INK, alpha=0.06, lw=0, zorder=2)
    ax.set_ylim(0, 62); ax.set_yticks([0, 20, 40, 60]); ax.set_yticklabels(["0", "20", "40", "60"] if j == 0 else [])
    S.year_axis(ax, every=2, start=2019); S.hgrid(ax); ax.tick_params(axis="y", length=0)
    chg = ends(ax, y)
    ttl = c if c != "Other" else f"{ncountry - 3} other countries"
    ax.annotate(ttl, xy=(0, 1), xycoords="axes fraction", xytext=(0, 15), textcoords="offset points", fontsize=7.6, fontweight="semibold", va="bottom")
    ax.annotate(chg, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", fontsize=6.6, color=S.INK2, va="bottom")
    if j == 0:
        ax.set_ylabel("Share of ads, %"); S.panel(ax, "a", "Country of the ad", dy=30)
ax = fig.add_subplot(top[0, 5])
y = stable.to_numpy()
ax.plot(S.MDATE, y, color=S.INK, lw=1.0, zorder=3); ax.fill_between(S.MDATE, 0, y, color=S.INK, alpha=0.06, lw=0, zorder=2)
ax.set_ylim(0, 93); ax.set_yticks([0, 30, 60, 90]); S.year_axis(ax, every=2, start=2019); S.hgrid(ax); ax.tick_params(axis="y", length=0)
chg = ends(ax, y)
ax.annotate("Sources flagged stable", xy=(0, 1), xycoords="axes fraction", xytext=(0, 15), textcoords="offset points", fontsize=7.6, fontweight="semibold", va="bottom")
ax.annotate(chg, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", fontsize=6.6, color=S.INK2, va="bottom")
S.panel(ax, "b", "Source of the ad", dy=30)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=S.INK, lw=1.0, label="Monthly share"), Line2D([], [], color=S.ACCENT, lw=1.6, label="Mean of the first 12 and of the last 12 months (the two numbers under each title)")],
           loc="upper left", bbox_to_anchor=(0.065, 1.0), ncol=2, fontsize=6.6, handlelength=1.6, columnspacing=1.6, borderaxespad=0)

# ---- c: agreement of each re-run with the pooled result ------------------------------------------------
agr = S.tab("tab10_stage_robustness_summary.csv").set_index("variant")
ROWS = [("United Kingdom", "United Kingdom only", "Country"), ("DEUTSCHLAND", "Germany only", "Country"), ("France", "France only", "Country"),
        ("Other", f"The {ncountry - 3} other countries", "Country"), ("Fixed mix", "All countries, fixed country weights", "Country"),
        ("src:stable", "Stable sources only", "Source"), ("src:unstable", "Unstable sources only", "Source"),
        ("No comp.prog.", "Without “computer programming”", "Basket")]
ax = fig.add_subplot(bot[0, 0])
pos = ax.get_position(); ax.set_position([0.44, pos.y0, 0.985 - 0.44 - 0.135, pos.height])
yy, ylab, prev = [], [], None
y = 0.0
for key, lab, g in ROWS:
    if prev is not None and g != prev:
        y += 0.55
    r = agr.loc[key]; n, same, opp = int(r["pairs_classified"]), int(r["same_stage_as_All"]), int(r["opposite_direction"])
    ax.barh(y, same / n * 100, height=0.62, color=S.INK2, lw=0, zorder=3)
    ax.barh(y, (n - same) / n * 100, left=same / n * 100 + 0.35, height=0.62, color=S.STAGE_COL["Maturity"], lw=0, zorder=3)
    ax.text(same / n * 100 - 1.2, y, f"{same / n:.0%}", ha="right", va="center", fontsize=6.8, color="white", fontweight="semibold", zorder=4)
    ax.text(102, y, f"{same} of {n} pairs" + (f"; {opp} opposite" if opp else ""), ha="left", va="center", fontsize=6.6, color=S.INK2, clip_on=False)
    ax.text(-2, y, lab, ha="right", va="center", fontsize=7.0, color=S.INK, clip_on=False)
    if g != prev:
        ax.text(-91, y, {"Country": "By country", "Source": "By source", "Basket": "Other basket"}[g], ha="left", va="center", fontsize=6.6,
                color=S.MUTED, fontweight="semibold", clip_on=False)
    prev = g; y += 1.0
ax.set_ylim(y - 0.45, -0.6); ax.set_xlim(0, 100.4); ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
ax.spines["left"].set_visible(False); ax.xaxis.grid(True, color=S.GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="x", length=0)
ax.set_xlabel("Pairs with the same stage as in the pooled data")
ax.annotate("c", xy=(0.065, pos.y1), xycoords="figure fraction", xytext=(0, 8), textcoords="offset points", fontsize=9, fontweight="bold", va="bottom")
ax.annotate("Same stage rule, re-run on parts of the data", xy=(0.065, pos.y1), xycoords="figure fraction", xytext=(11, 8.6), textcoords="offset points",
            fontsize=8, fontweight="semibold", va="bottom")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=S.INK2, label="Same stage"), Patch(facecolor=S.STAGE_COL["Maturity"], label="Different stage")],
          loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=6.6, handlelength=1.0, handleheight=0.8, columnspacing=1.4, borderaxespad=0)
S.save(fig, "fig9_country_source")
print(ncountry, share.groupby(share.index.str[:4]).mean().round(1).to_string()); print(stable.groupby(stable.index.str[:4]).mean().round(1).to_dict())
