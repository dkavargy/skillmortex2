"""Fig. 1  The daily export cap: what was retrieved, what it leaves intact, and what it distorts."""
import calendar
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import smxfig as S
S.style()

sk, tt, prog = S.base_tables()

# ---- rows retrieved per day and per ad, by occupation and month (from the file audit) ----------------
a = S.tab("audit_files.csv")
fn = a["file"].str.split("__").str[-1].str.replace(".csv", "")
low = fn.str.lower()
a["occ"] = np.select([low.str.contains("software"), low.str.contains("systems"), low.str.contains("web"), low.str.contains("applications")],
                     S.OCCUPATIONS, "?")
a["ym"] = fn.str.extract(r"(\d{4}-\d{2})")[0]
def ndays(f, ym):
    if pd.isna(ym):
        return np.nan
    y, m = map(int, ym.split("-")); d = calendar.monthrange(y, m)[1]
    return 15 if "_1_15" in f else (d - 15 if "_16_end" in f else d)
a["days"] = [ndays(f, y) for f, y in zip(fn, a["ym"])]
mon = a.dropna(subset=["ym"]).groupby(["occ", "ym"], as_index=False).agg(rows=("rows", "sum"), postings=("postings", "sum"))
mon["days"] = [calendar.monthrange(*map(int, y.split("-")))[1] for y in mon["ym"]]
mon["rows_day"] = mon["rows"] / mon["days"]
mon["rows_ad"] = mon["rows"] / mon["postings"]
half = a[a["ym"].isna()].copy()                            # half-year files (Applications programmers, 2018-2020)
hy = fn[half.index].str.extract(r"(\d{4})_H([12])")
extra = []
for (i, r), (y, h) in zip(half.iterrows(), hy.itertuples(index=False)):
    ms = [f"{y}-{m:02d}" for m in (range(1, 7) if h == "1" else range(7, 13))]
    nd = sum(calendar.monthrange(int(y), int(m[-2:]))[1] for m in ms)
    extra += [{"occ": r["occ"], "ym": m, "rows_day": r["rows"] / nd, "rows_ad": r["rows"] / r["postings"]} for m in ms]
mon = pd.concat([mon, pd.DataFrame(extra)], ignore_index=True)

fig = plt.figure(figsize=(7.2, 5.15))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.3], hspace=0.66)
top = gs[0].subgridspec(1, 4, wspace=0.12)
bot = gs[1].subgridspec(1, 3, wspace=0.34, width_ratios=[1.0, 1.0, 1.0])

# ---- a: rows per day -------------------------------------------------------------------------------
note = {}
for j, occ in enumerate(S.OCCUPATIONS):
    ax = fig.add_subplot(top[0, j])
    g = mon[(mon["occ"] == occ) & (mon["ym"] >= S.START_MONTH)].sort_values("ym")
    x = pd.to_datetime(g["ym"] + "-01")
    for capv in (10000, 15000):
        ax.axhline(capv, color=S.AXIS, lw=0.6, zorder=1)
    ax.plot(x, g["rows_day"], color=S.INK, lw=1.3, zorder=3)
    ax.fill_between(x, 0, g["rows_day"], color=S.INK, alpha=0.06, lw=0, zorder=2)
    ax.set_ylim(0, 17500); ax.set_yticks([0, 5000, 10000, 15000])
    ax.set_yticklabels(["0", "5,000", "10,000", "15,000"] if j == 0 else [])
    S.year_axis(ax, every=2, start=2020)
    ax.tick_params(axis="y", length=0)
    atcap = float((((g["rows_day"] >= 9700) & (g["rows_day"] <= 10001)) | ((g["rows_day"] >= 14500) & (g["rows_day"] <= 15001))).mean())
    ra = g["rows_ad"]
    ax.annotate(S.OCC_SHORT[occ], xy=(0, 1), xycoords="axes fraction", xytext=(0, 22), textcoords="offset points",
                fontsize=7.8, fontweight="semibold", color=S.INK, va="bottom", ha="left")
    sub = (f"at the cap in {atcap:.0%} of months" if atcap > 0 else "below the cap in every month")
    ax.annotate(sub + f"\n{ra.quantile(.1):.0f}\u2013{ra.quantile(.9):.0f} rows retrieved per ad", xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 3), textcoords="offset points", fontsize=6.4, color=S.INK2, va="bottom", ha="left", linespacing=1.3)
    if j == 0:
        ax.set_ylabel("Rows retrieved per day\n(monthly mean)")
        S.panel(ax, "a", "Rows retrieved per day, by occupation", dy=38)
        ax.annotate("10,000-row cap", xy=(pd.Timestamp("2022-01-01"), 10000), xytext=(0, 3), textcoords="offset points",
                    fontsize=6.2, color=S.MUTED, ha="center", va="bottom")
    if j == 1:
        ax.annotate("15,000-row cap", xy=(pd.Timestamp("2020-10-01"), 15000), xytext=(0, 3), textcoords="offset points",
                    fontsize=6.2, color=S.MUTED, ha="center", va="bottom")
    note[occ] = (atcap, ra.median())

# ---- b: skill mix with and without the cap (Systems analysts) -----------------------------------------
ax = fig.add_subplot(bot[0, 0])
Sa = sk[sk["occupation4d"] == "Systems analysts"]
def prof(ms):
    x = Sa[Sa["month"].isin(ms)].groupby("skill")["n_post"].sum()
    return x / x.sum() * 100
u, c = prof(["2018-08", "2018-09"]), prof(["2018-10", "2018-11", "2018-12"])
df = pd.DataFrame({"u": u, "c": c}).fillna(0)
df = df[(df["u"] > 0.05) & (df["c"] > 0)]
r = np.corrcoef(np.log(df["u"]), np.log(df["c"]))[0, 1]
lo, hi = 0.04, 6
ax.plot([lo, hi], [lo, hi], color=S.AXIS, lw=0.7, zorder=1)
isp = df.index.isin(S.PROG_SKILLS)
ax.scatter(df.loc[~isp, "u"], df.loc[~isp, "c"], s=9, color=S.MUTED, alpha=0.55, lw=0, zorder=2)
ax.scatter(df.loc[isp, "u"], df.loc[isp, "c"], s=20, color=S.ACCENT, edgecolor="white", lw=0.7, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(plt.FixedLocator([0.1, 1])); axis.set_major_formatter(plt.FixedFormatter(["0.1%", "1%"]))
    axis.set_minor_locator(plt.NullLocator())
ax.set_xlabel("Share of skill mentions, no cap\n(Aug–Sep 2018)"); ax.set_ylabel("Share of skill mentions, cap\n(Oct–Dec 2018)")
ax.annotate(f"{len(df)} skills\nr = {r:.2f}", xy=(0.06, 0.94), xycoords="axes fraction", fontsize=7, color=S.INK, va="top", linespacing=1.3)
ax.scatter([], [], s=20, color=S.ACCENT, edgecolor="white", lw=0.7, label="The 13 programming skills")
ax.scatter([], [], s=9, color=S.MUTED, alpha=0.55, lw=0, label="Other skills")
ax.legend(loc="lower right", fontsize=6.2, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.3)
S.panel(ax, "b", "Skill mix, cap vs no cap", dy=18)
ax.annotate("Systems analysts, 2018", xy=(0, 1), xycoords="axes fraction", xytext=(0, 4),
            textcoords="offset points", fontsize=6.4, color=S.INK2, ha="left", va="bottom")

# ---- c, d: one occupation, all skills, two measures ------------------------------------------------
occ = "Software developers"
base = [m for m in S.MLIST if m.startswith("2023")]
def indexed(col):
    out = {}
    for skl in S.PROG_SKILLS:
        s, cnt = S.series(skl, occ, col)
        if (cnt > 0).sum() < S.MIN_MONTHS:
            continue
        sm = pd.Series(s, index=S.MLIST).rolling(6, min_periods=6).mean()
        out[skl] = sm / sm[base].mean() * 100
    return pd.DataFrame(out)
ra = (sk[sk["occupation4d"] == occ].groupby("month")["n_mentions"].sum() / tt[tt["occupation4d"] == occ].set_index("month")["den"]).reindex(S.MLIST)
ra = ra.rolling(6, min_periods=6).mean(); ra = ra / ra[base].mean() * 100

for i, (col, ttl, letter, subt) in enumerate([
        ("post", "Share of ads", "c", "Every skill follows the rows retrieved per ad"),
        ("rel", "Relative share", "d", "The common movement cancels out")]):
    ax = fig.add_subplot(bot[0, i + 1])
    I = indexed(col)
    ax.axhline(100, color=S.AXIS, lw=0.6, zorder=1)
    for skl in I.columns:
        ax.plot(S.MDATE, I[skl], color=S.MUTED, lw=0.7, alpha=0.75, zorder=2)
    ax.plot(S.MDATE, I.median(axis=1), color=S.INK, lw=1.8, zorder=4)
    if col == "post":
        ax.plot(S.MDATE, ra, color=S.ACCENT, lw=1.8, zorder=3)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=S.INK, lw=1.8, label="Median skill"),
                           Line2D([], [], color=S.ACCENT, lw=1.8, label="Rows retrieved per ad")],
                  loc="upper center", bbox_to_anchor=(0.5, 0.97), fontsize=6.3, handlelength=1.3, labelspacing=0.3, borderaxespad=0.1)
    else:
        import matplotlib.patheffects as pe
        ax.annotate("Median skill", xy=(S.MDATE[40], 100), xytext=(0, 26), textcoords="offset points", fontsize=6.6, color=S.INK2,
                    ha="center", va="bottom", arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.5, shrinkA=1, shrinkB=2),
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_ylim(40, 228); ax.set_yticks([50, 100, 150, 200])
    S.year_axis(ax, every=1, start=2020); ax.set_xlim(S.MDATE[5], S.MDATE[-1])
    S.hgrid(ax); ax.tick_params(axis="y", length=0)
    ax.set_ylabel("Index, 2023 = 100" if i == 0 else "")
    S.panel(ax, letter, ttl, dy=18)
    ax.annotate(subt, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=6.4, color=S.INK2, ha="left", va="bottom")
    ax.annotate(f"{S.OCC_SHORT[occ]}, {I.shape[1]} skills, 6-month mean", xy=(1, 0), xycoords="axes fraction", xytext=(0, -20),
                textcoords="offset points", fontsize=6.2, color=S.MUTED, ha="right", va="top")

S.save(fig, "fig1_export_cap")
print({S.OCC_ABBR[k]: (round(v[0], 2), round(v[1], 1)) for k, v in note.items()}, "| r", round(r, 4), "| n", len(df))
