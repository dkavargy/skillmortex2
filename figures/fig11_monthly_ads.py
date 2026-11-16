"""Fig. 11  Advertisements retrieved per month, four occupations in one panel (redraw of the old monthly-postings figure).
The daily export cap is reported in the data table and in Figure 1, not here."""
import calendar, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import smxfig as S
S.style()
sk, tt, prog = S.base_tables()
a = S.tab("audit_files.csv"); fn = a["file"].str.split("__").str[-1].str.replace(".csv", ""); low = fn.str.lower()
a["occ"] = np.select([low.str.contains("software"), low.str.contains("systems"), low.str.contains("web"), low.str.contains("applications")], S.OCCUPATIONS, "?")
a["ym"] = fn.str.extract(r"(\d{4}-\d{2})")[0]
mon = a.dropna(subset=["ym"]).groupby(["occ", "ym"], as_index=False)["rows"].sum()
mon["rows_day"] = mon["rows"] / [calendar.monthrange(*map(int, y.split("-")))[1] for y in mon["ym"]]
capped = {(r.occ, r.ym) for r in mon.itertuples() if (9700 <= r.rows_day <= 10001) or (14500 <= r.rows_day <= 15001)}
COL = dict(zip(S.OCCUPATIONS, S.IDENT))                           # same occupation colours as Figure 2
half = pd.Timedelta(days=15)

fig = plt.figure(figsize=(7.2, 3.4))
ax = fig.add_axes([0.075, 0.085, 0.74, 0.80])
ends = {}
for occ in S.OCCUPATIONS:
    y = tt[tt["occupation4d"] == occ].set_index("month")["den"].reindex(S.MLIST).to_numpy(float) / 1000
    ax.plot(S.MDATE, y, color=COL[occ], lw=1.6, zorder=3, solid_joinstyle="round", path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ends[occ] = y[-1]
ax.set_ylim(0, 128); ax.set_yticks([0, 25, 50, 75, 100, 125]); S.hgrid(ax); ax.tick_params(axis="y", length=0)
ax.set_xlim(S.MDATE[0] - half, S.MDATE[-1] + half); ax.set_ylabel("Advertisements retrieved per month (thousand)")
# direct labels at the line ends, spread so they do not collide
ypos = dict(ends); order = sorted(S.OCCUPATIONS, key=lambda o: ends[o]); gap = 7.5
for a_, b_ in zip(order[:-1], order[1:]):
    if ypos[b_] - ypos[a_] < gap:
        ypos[b_] = ypos[a_] + gap
for occ in S.OCCUPATIONS:
    ax.plot([S.MDATE[-1] + half * 1.3, S.MDATE[-1] + half * 3.0], [ends[occ], ypos[occ]], color=COL[occ], lw=0.9, clip_on=False, solid_capstyle="round")
    ax.annotate(S.OCC_SHORT[occ], xy=(S.MDATE[-1] + half * 3.4, ypos[occ]), fontsize=7.2, color=S.INK, va="center", ha="left", annotation_clip=False)
ax.legend(handles=[Line2D([], [], color=COL[o], lw=1.6, label=S.OCC_SHORT[o]) for o in S.OCCUPATIONS], loc="lower left", bbox_to_anchor=(0, 1.02),
          ncol=4, fontsize=6.8, handlelength=1.5, columnspacing=1.6, borderaxespad=0)

S.year_axis(ax, every=1, start=2019, short=False)
S.save(fig, "fig11_monthly_ads")
