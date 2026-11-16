"""Fig. 10  Where the advertisements come from: share by country, October 2018 to March 2025 (redraw of the old map)."""
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
import smxfig as S
S.style()
ISO = {"United Kingdom": ("GBR", "United Kingdom"), "DEUTSCHLAND": ("DEU", "Germany"), "France": ("FRA", "France"), "Nederland": ("NLD", "Netherlands"),
       "ITALIA": ("ITA", "Italy"), "Polska": ("POL", "Poland"), "ESPAÑA": ("ESP", "Spain"), "Belgique/België": ("BEL", "Belgium"), "Sverige": ("SWE", "Sweden"),
       "Österreich": ("AUT", "Austria"), "Portugal": ("PRT", "Portugal"), "Éire/Ireland": ("IRL", "Ireland"), "Česko": ("CZE", "Czechia"), "ROMÂNIA": ("ROU", "Romania"),
       "Schweiz/Suisse/Svizzera": ("CHE", "Switzerland"), "България": ("BGR", "Bulgaria"), "Magyarország": ("HUN", "Hungary"), "Slovensko": ("SVK", "Slovakia"),
       "Lietuva": ("LTU", "Lithuania"), "Danmark": ("DNK", "Denmark"), "Ελλάδα": ("GRC", "Greece"), "Luxembourg": ("LUX", "Luxembourg"), "Suomi/Finland": ("FIN", "Finland"),
       "Latvija": ("LVA", "Latvia"), "Norge": ("NOR", "Norway"), "Eesti": ("EST", "Estonia"), "Κύπρος": ("CYP", "Cyprus"), "Hrvatska": ("HRV", "Croatia"), "Malta": ("MLT", "Malta"),
       "Slovenija": ("SVN", "Slovenia"), "Liechtenstein": ("LIE", "Liechtenstein"), "Ísland": ("ISL", "Iceland")}
t = S.rd("totals_country"); t = t[t["month"].isin(set(S.MLIST))].copy(); t["country"] = t["country"].str.replace("\u00a0", "").str.strip()
g = t.groupby("country")["n_post"].sum(); assert set(g.index) <= set(ISO), set(g.index) - set(ISO)
share = pd.DataFrame({"a3": [ISO[c][0] for c in g.index], "name": [ISO[c][1] for c in g.index], "n": g.to_numpy()})
share["pct"] = share["n"] / share["n"].sum() * 100; share = share.sort_values("pct", ascending=False).reset_index(drop=True)
pct = share.set_index("a3")["pct"]

BINS = [0, 0.5, 1, 2.5, 5, 10, 20, 100]
RAMP = ["#fff5c2", "#fee08a", "#fdbb55", "#f98f3a", "#ea5a2b", "#c4282a", "#7d0f24"]            # yellow, orange, red, burgundy
LAB = ["under 0.5", "0.5–1", "1–2.5", "2.5–5", "5–10", "10–20", "over 20"]
col = lambda v: RAMP[int(np.searchsorted(BINS, v, side="right") - 1)]

def laea(lon, lat, lon0=10.0, lat0=52.0):
    lam, phi, l0, p0 = np.radians(lon), np.radians(lat), np.radians(lon0), np.radians(lat0)
    k = np.sqrt(2 / (1 + np.sin(p0) * np.sin(phi) + np.cos(p0) * np.cos(phi) * np.cos(lam - l0)))
    return k * np.cos(phi) * np.sin(lam - l0), k * (np.cos(p0) * np.sin(phi) - np.sin(p0) * np.cos(phi) * np.cos(lam - l0))

geo = json.load(open(os.path.join(S.HERE, "..", "geo", "europe_50m.json")))
fig = plt.figure(figsize=(7.2, 4.9))
ax = fig.add_axes([0.0, 0.0, 0.60, 1.0]); ax.set_aspect("equal"); ax.axis("off")
for f in sorted(geo, key=lambda f: f["a3"] in pct.index):                       # countries with data drawn last, on top
    has = f["a3"] in pct.index
    verts, codes = [], []
    for poly in f["polys"]:
        for ring in poly:
            a = np.array(ring); x, y = laea(a[:, 0], a[:, 1])
            verts += list(zip(x, y)); codes += [Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=col(pct[f["a3"]]) if has else "#f4f3ef", edgecolor="#4a4945" if has else "#b9b8ae",
                           lw=0.55 if has else 0.4, joinstyle="round", zorder=2 if has else 1))
x0, y0 = laea(-11.5, 34.5); x1, y1 = laea(33.5, 70.5)
ax.set_xlim(x0 - 0.13, x1 + 0.10); ax.set_ylim(y0 - 0.03, y1 + 0.045)
for f in geo:                                                                    # value on the map for the larger countries
    if f["a3"] in pct.index and pct[f["a3"]] >= 2.0:
        v = pct[f["a3"]]; x, y = laea(f["lx"], f["ly"])
        ax.text(x, y, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center", fontsize=7.4 if v >= 10 else 6.4, fontweight="semibold",
                color="white" if v >= 10 else S.INK, zorder=5)
# legend: binned classes, stacked in the Atlantic
lx, ly, w, h = 0.022, 0.535, 0.024, 0.024
fig.text(lx, ly + h + 0.036, "Share of\nadvertisements, %", fontsize=7.0, fontweight="semibold", color=S.INK, va="center", linespacing=1.25)
for i, (c, lab) in enumerate(zip(RAMP[::-1], LAB[::-1])):
    yy = ly - i * (h + 0.004)
    fig.patches.append(Rectangle((lx, yy), w, h, transform=fig.transFigure, facecolor=c, edgecolor="#4a4945", lw=0.4))
    fig.text(lx + w + 0.008, yy + h / 2, lab, fontsize=6.2, color=S.INK2, ha="left", va="center")
yy = ly - 7 * (h + 0.004) - 0.012
fig.patches.append(Rectangle((lx, yy), w, h, transform=fig.transFigure, facecolor="#f4f3ef", edgecolor="#b9b8ae", lw=0.4))
fig.text(lx + w + 0.008, yy + h / 2, "not in the data", fontsize=6.2, color=S.INK2, ha="left", va="center")

# ranked bars
TOP = 14
bx = fig.add_axes([0.745, 0.085, 0.215, 0.80])
rows = [(r["name"], r["pct"]) for _, r in share.head(TOP).iterrows()] + [(f"{len(share) - TOP} other countries", share["pct"].iloc[TOP:].sum())]
for i, (nm, v) in enumerate(rows):
    last = i == len(rows) - 1; yy = i + (0.45 if last else 0)
    bx.barh(yy, v, height=0.62, color="#f4f3ef" if last else col(v), edgecolor="#b9b8ae" if last else "#4a4945", lw=0.4, zorder=3)
    bx.text(-0.8, yy, nm, ha="right", va="center", fontsize=7.0, color=S.INK, clip_on=False)
    bx.text(v + 0.6, yy, f"{v:.1f}", ha="left", va="center", fontsize=6.8, color=S.INK2, clip_on=False)
bx.set_ylim(len(rows) + 0.1, -0.7); bx.set_xlim(0, 30); bx.set_yticks([]); bx.set_xticks([0, 10, 20, 30]); bx.set_xticklabels(["0", "10", "20", "30%"])
bx.spines["left"].set_visible(False); bx.xaxis.grid(True, color=S.GRID, lw=0.5); bx.set_axisbelow(True); bx.tick_params(axis="x", length=0)
bx.set_title("Share of advertisements", fontsize=7.6, fontweight="semibold", loc="left", pad=6, x=-0.62)
S.save(fig, "fig10_country_map")
print(share.round(2).to_string()); print("total ads", int(share["n"].sum()), "countries", len(share))
