"""Fig. 8  What the old rapid-obsolescence rule detects: every flagged skill, with the level before and after its peak."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import smxfig as S
S.style()
sk, tt, prog = S.base_tables()
aud = S.tab("tab9_rapid_obsolescence_audit.csv")
fl = aud[aud["flagged_by_old_rule"] == "yes"].copy()
pool = sk[sk["month"].isin(set(S.MLIST))].groupby(["month", "skill"], as_index=False)["n_post"].sum()
pool["share"] = pool["n_post"] / pool.groupby("month")["n_post"].transform("sum") * 100
READ = {"spike: level afterwards is back to before": ("Spike, level back", 0), "level fell": ("Level fell", 1),
        "too close to an end of the data": ("Too recent to tell", 2)}
fl["grp"] = fl["reading"].map(lambda r: READ[r][1]); fl = fl.sort_values(["grp", "peak_month"])
e = S.MLIST.index(S.EVENT)

fig = plt.figure(figsize=(7.2, 5.7))
gs = GridSpec(3, 4, figure=fig, left=0.055, right=0.985, top=0.825, bottom=0.06, hspace=1.02, wspace=0.26)
for n, r in enumerate(fl.itertuples()):
    ax = fig.add_subplot(gs[n // 4, n % 4])
    s = pool[pool["skill"] == r.skill].set_index("month")["share"].reindex(S.MLIST, fill_value=0.0).to_numpy()
    ip = int(s.argmax())
    ax.axvline(S.MDATE[e], color=S.AXIS, lw=0.7, zorder=1)
    ax.plot(S.MDATE, s, color=S.INK, lw=1.0, zorder=3)
    ax.scatter([S.MDATE[ip]], [s[ip]], s=14, color=S.INK, edgecolor="white", lw=0.7, zorder=4)
    if ip - 6 >= 6:                                             # level before: months 18 to 7 before the peak
        b0, b1 = max(ip - 18, 0), ip - 6; bef = np.median(s[b0:b1])
        ax.plot([S.MDATE[b0], S.MDATE[b1 - 1]], [bef, bef], color=S.ACCENT, lw=2.2, solid_capstyle="butt", zorder=5)
    if ip + 12 < len(s):                                        # level after: months 7 to 18 after the peak
        a0, a1 = ip + 7, min(ip + 19, len(s)); aft = np.median(s[a0:a1])
        ax.plot([S.MDATE[a0], S.MDATE[a1 - 1]], [aft, aft], color=S.STAGE_COL["Decline"], lw=2.2, solid_capstyle="butt", zorder=5)
    ax.set_ylim(0, s.max() * 1.15); S.year_axis(ax, every=2, start=2019)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, steps=[1, 2, 2.5, 5, 10])); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="y", length=0, labelsize=6.2); ax.tick_params(axis="x", labelsize=6.2); S.hgrid(ax)
    name = r.skill[0].upper() + r.skill[1:]
    name = {"Abap": "ABAP", "Ict communications protocols": "ICT communications protocols", "Microsoft access": "Microsoft Access",
            "Identify customer's needs": "Identify customer\u2019s needs"}.get(name, name)
    ax.annotate(name, xy=(0, 1), xycoords="axes fraction", xytext=(0, 23), textcoords="offset points", fontsize=7.2, fontweight="semibold", va="bottom")
    pk = pd.Period(r.peak_month, "M").strftime("%b %Y")
    ratio = "no 12 months after it yet" if pd.isna(r.level_after_vs_before) else f"after \u00f7 before = {r.level_after_vs_before:.2f}"
    ax.annotate(f"Peak {pk}\n{ratio}", xy=(0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                fontsize=6.0, color=S.INK2, va="bottom", linespacing=1.3)
    if n in (0, 5, 9):                                          # group heading above the first panel of each reading
        ax.annotate({0: "Spike: the level afterwards is back to the level before", 5: "The level fell", 9: "Too recent to tell"}[n],
                    xy=(0, 1), xycoords="axes fraction", xytext=(0, 37), textcoords="offset points", fontsize=6.6, color=S.MUTED,
                    va="bottom", fontweight="semibold")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=S.INK, lw=1.0, label="Monthly share of all skill mentions (%)"),
                    Line2D([], [], color=S.ACCENT, lw=2.2, label="Level before the peak"),
                    Line2D([], [], color=S.STAGE_COL["Decline"], lw=2.2, label="Level after the peak"),
                    Line2D([], [], color=S.AXIS, lw=0.7, label="November 2022")],
           loc="upper left", bbox_to_anchor=(0.055, 1.0), ncol=4, fontsize=6.6, columnspacing=1.8, handlelength=1.6, borderaxespad=0)
S.save(fig, "fig8_spike_audit")
print(fl[["skill", "reading", "peak_month", "level_after_vs_before"]].to_string(index=False))
