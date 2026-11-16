"""Fig. 6  Forecast evaluation: no model beats carrying the last value forward."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import smxfig as S
S.style()
ev = pd.read_parquet(S.DATA + "/eval_long_relative.parquet")
ev["ae"] = (ev["y"] - ev["yhat"]).abs()
ev["covered"] = ((ev["y"] >= ev["lo"]) & (ev["y"] <= ev["hi"])).astype(float)
ev["bucket"] = pd.cut(ev["h"], [0, 12, 24, 36], labels=["1–12 months ahead", "13–24 months ahead", "25–36 months ahead"])
PK = ["skill", "occupation"]
MODELS = [("SeasonalNaive", "Seasonal naive"), ("HW_Additive_Damped", "Holt-Winters, damped"), ("HW_Multiplicative", "Holt-Winters, multiplicative"),
          ("SARIMA(0,1,1)(0,1,1)12", "SARIMA (0,1,1)(0,1,1)"), ("SARIMA(1,1,1)(1,1,1)12", "SARIMA (1,1,1)(1,1,1)"), ("Prophet", "Prophet")]

# pair-level MAE per model and horizon bucket, then ratio to Naive on the same cells
pm = ev.groupby(PK + ["bucket", "model"], observed=True)["ae"].mean().unstack("model")
rel = pm.div(pm["Naive"], axis=0)

fig = plt.figure(figsize=(7.2, 4.9))
gs = GridSpec(1, 3, figure=fig, wspace=0.08, left=0.215, right=0.985, top=0.895, bottom=0.555)
gb = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1.0], wspace=0.62, left=0.075, right=0.985, top=0.385, bottom=0.085)
rng = np.random.default_rng(1)
summ = []
for j, b in enumerate(rel.index.get_level_values("bucket").categories):
    ax = fig.add_subplot(gs[0, j]); R = rel.xs(b, level="bucket")
    ax.axvline(1, color=S.INK, lw=0.8, zorder=1)
    for i, (m, lab) in enumerate(MODELS):
        v = R[m].dropna().to_numpy(); yj = i + rng.uniform(-0.24, 0.24, len(v))
        better = v < 1
        ax.scatter(np.clip(v[~better], 0.45, 5.0), yj[~better], s=7, color=S.MUTED, alpha=0.55, lw=0, zorder=2)
        ax.scatter(np.clip(v[better], 0.47, 5.0), yj[better], s=9, color=S.ACCENT, alpha=0.9, lw=0, zorder=3)
        med = np.median(v)
        ax.plot([med, med], [i - 0.36, i + 0.36], color=S.INK, lw=1.8, zorder=4, solid_capstyle="butt")
        ax.annotate(f"{better.mean():.0%}", xy=(1, i), xycoords=("axes fraction", "data"), xytext=(-1, 0), textcoords="offset points",
                    fontsize=6.4, color=S.INK2, ha="right", va="center")
        summ.append((b, lab, round(float(med), 2), round(float(better.mean()), 2), len(v)))
    ax.set_xscale("log"); ax.set_xlim(0.44, 8.5); ax.set_ylim(len(MODELS) - 0.45, -0.95)
    ax.xaxis.set_major_locator(plt.FixedLocator([0.5, 1, 2, 4])); ax.xaxis.set_major_formatter(plt.FixedFormatter(["0.5", "1", "2", "4"]))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([l for _, l in MODELS] if j == 0 else [], fontsize=7)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_title(b, fontsize=7.8, pad=14)
    ax.annotate("pairs\nbetter", xy=(1, -0.72), xycoords=("axes fraction", "data"), xytext=(-1, 0), textcoords="offset points",
                fontsize=5.6, color=S.MUTED, ha="right", va="center", linespacing=1.1)
    if j == 0:
        S.panel(ax, "a", "Error of each model for each skill\u2013occupation pair", dy=27)
    ax.annotate("better", xy=(1, -0.72), xytext=(-4, 0), textcoords="offset points", fontsize=5.9, color=S.INK2, ha="right", va="center")
    ax.annotate("worse than naive", xy=(1, -0.72), xytext=(4, 0), textcoords="offset points", fontsize=5.9, color=S.INK2, ha="left", va="center")
    if j == 1:
        ax.set_xlabel("Mean absolute error relative to naive (last value carried forward), log scale", labelpad=3)

# ---- b: error by forecast step, relative to naive ------------------------------------------------------
axb = fig.add_subplot(gb[0, 0])
st = ev.groupby(PK + ["model", "h"], observed=True)["ae"].mean().unstack("model")
relh = st.div(st["Naive"], axis=0).groupby("h").median()
axb.axhline(1, color=S.INK, lw=0.8, zorder=1)
from matplotlib.lines import Line2D
for m, lab in MODELS:
    hi = m in ("SeasonalNaive", "HW_Additive_Damped")
    axb.plot(relh.index, relh[m], color=S.INK if m == "HW_Additive_Damped" else (S.ACCENT if m == "SeasonalNaive" else S.MUTED),
             lw=1.7 if hi else 0.8, alpha=1 if hi else 0.8, zorder=4 if hi else 2)
axb.set_xlim(1, 36); axb.set_xticks([1, 6, 12, 18, 24, 30, 36]); axb.set_ylim(0.82, 3.15); axb.set_yticks([1, 1.5, 2, 2.5, 3])
S.hgrid(axb); axb.tick_params(axis="y", length=0)
axb.set_xlabel("Months ahead"); axb.set_ylabel("Median error relative to naive")
axb.legend(handles=[Line2D([], [], color=S.INK, lw=1.7, label="Holt-Winters, damped"), Line2D([], [], color=S.ACCENT, lw=1.7, label="Seasonal naive"),
                    Line2D([], [], color=S.MUTED, lw=0.8, label="Other models")], loc="upper left", fontsize=6.4, handlelength=1.4,
           labelspacing=0.3, borderaxespad=0.3)
axb.annotate("Naive = 1", xy=(1.4, 1), xytext=(0, -3), textcoords="offset points", fontsize=6.2, color=S.INK2, ha="left", va="top")
S.panel(axb, "b", "Error by forecast step", dy=18)

# ---- c: coverage of the 80% intervals ------------------------------------------------------------------
axc = fig.add_subplot(gb[0, 1])
cov = ev.groupby(["model", "bucket"], observed=True)["covered"].mean().unstack("bucket") * 100
rows = [("Naive", "Naive")] + MODELS
axc.axvline(80, color=S.INK, lw=0.8, zorder=1)
for i, (m, lab) in enumerate(rows):
    v = cov.loc[m].to_numpy()
    axc.plot([v.min(), v.max()], [i, i], color=S.AXIS, lw=1.2, zorder=2, solid_capstyle="round")
    axc.scatter(v, [i] * 3, s=[14, 22, 32], color=S.INK2, edgecolor="white", lw=0.7, zorder=3)
axc.set_ylim(len(rows) - 0.5, -0.6); axc.set_yticks(range(len(rows))); axc.set_yticklabels([l.replace("Holt-Winters", "HW").replace(", multiplicative", ", mult.") for _, l in rows], fontsize=6.4)
axc.set_xlim(10, 104); axc.set_xticks([20, 40, 60, 80, 100]); axc.set_xticklabels(["20", "40", "60", "80", "100%"])
axc.tick_params(axis="y", length=0); axc.spines["left"].set_visible(False)
axc.set_xlabel("Outcomes inside the 80% interval")
axc.annotate("nominal", xy=(80, 0), xycoords=("data", "axes fraction"), xytext=(-3, 3), textcoords="offset points", fontsize=6.0, color=S.INK2, ha="right", va="bottom")
S.panel(axc, "c", "Interval coverage", dy=18)
axc.annotate("Dots: the three horizons, small to large", xy=(0, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
             fontsize=6.2, color=S.INK2, ha="left", va="bottom")
S.save(fig, "fig6_forecast_eval")
print(pd.DataFrame(summ, columns=["bucket", "model", "median_relMAE", "share_better", "pairs"]).to_string(index=False))
print(cov.round(0).to_string())
