"""Shared style, palette, data loading and the stage rule for the SkillMORTEX 2.0 revision figures.

Every figure script imports this module, so colours, type and the stage rule are defined once.
The stage rule is the same code that produced tab5b_stages_relative_share.csv.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SMX_DATA", os.path.join(HERE, "..", "data", "smx_r1"))
OUT  = os.environ.get("SMX_FIGOUT", os.path.join(HERE, "..", "out"))
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- palette (validated, see README) -----
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

STAGE_COL = {            # Growth / Decline / Revival pass the colour-blind checks on all pairs
    "Growth":   "#2a78d6",
    "Maturity": "#c9c8bf",      # neutral: the common, quiet state
    "Decline":  "#eb6834",
    "Revival":  "#1baf7a",
    "Death":    "#2b2b29",
    "Birth":    "#eda100",
}
STAGE_ORDER = ["Growth", "Maturity", "Decline", "Revival", "Death"]
IDENT = ["#6250d6", "#e87ba4", "#008300", "#eda100"]          # identity series, always directly labelled
ACCENT = "#2a78d6"

PROG_SKILLS = [
    "computer programming", "java (computer programming)", "sql", "php",
    "web programming", "use object-oriented programming",
    "use scripting programming", "python (computer programming)", "css",
    "typescript", "object-oriented modelling", "ict system programming", "abap",
]
SKILL_LABEL = {
    "computer programming": "Computer programming", "java (computer programming)": "Java", "sql": "SQL", "php": "PHP",
    "web programming": "Web programming", "use object-oriented programming": "Object-oriented programming",
    "use scripting programming": "Scripting", "python (computer programming)": "Python", "css": "CSS",
    "typescript": "TypeScript", "object-oriented modelling": "Object-oriented modelling",
    "ict system programming": "ICT system programming", "abap": "ABAP",
}
OCCUPATIONS = ["Software developers", "Systems analysts", "Web and multimedia developers", "Applications programmers"]
OCC_SHORT = {"Software developers": "Software developers", "Systems analysts": "Systems analysts",
             "Web and multimedia developers": "Web developers", "Applications programmers": "Applications programmers"}
OCC_ABBR = {"Software developers": "SW dev", "Systems analysts": "Sys analysts",
            "Web and multimedia developers": "Web dev", "Applications programmers": "App prog"}

START_MONTH, END_MONTH, EVENT = "2018-10", "2025-03", "2022-11"
MONTHS = pd.period_range(START_MONTH, END_MONTH, freq="M")
MLIST = list(MONTHS.astype(str))
MDATE = MONTHS.to_timestamp()

# ---------------------------------------------------------------- style -------------------------------
def style():
    """Inter if it is installed, otherwise matplotlib's default sans. TrueType copies in SMX_FONTS (or ../fonts_ttf) are
    preferred, because matplotlib embeds TrueType cleanly in PDF and OpenType-CFF less so."""
    fm = font_manager.fontManager
    ttf_dir = os.environ.get("SMX_FONTS", os.path.join(HERE, "..", "fonts_ttf"))
    ttfs = [os.path.join(ttf_dir, f) for f in sorted(os.listdir(ttf_dir)) if f.lower().endswith(".ttf")] if os.path.isdir(ttf_dir) else []
    if ttfs:
        fm.ttflist = [f for f in fm.ttflist if not (f.name == "Inter" and not f.fname.lower().endswith(".ttf"))]
        for f in ttfs:
            fm.addfont(f)
    else:
        for f in font_manager.findSystemFonts():
            if os.path.basename(f).startswith("Inter-"):
                try:
                    fm.addfont(f)
                except Exception:
                    pass
    have_inter = any(f.name == "Inter" for f in fm.ttflist)
    mpl.rcParams.update({
        "font.family": "Inter" if have_inter else "sans-serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.4, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "semibold",
        "axes.titlepad": 5, "hatch.linewidth": 0.5, "hatch.color": AXIS,
    })

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved", name)

def panel(ax, letter, title=None, x=0.0, y=1.0, dy=6):
    """Panel letter (bold) and optional title, placed above the axes."""
    txt = ax.annotate(letter, xy=(x, y), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                      fontsize=9, fontweight="bold", color=INK, va="bottom", ha="left")
    if title:
        ax.annotate(title, xy=(x, y), xycoords="axes fraction", xytext=(11, dy + 0.6), textcoords="offset points",
                    fontsize=8, fontweight="semibold", color=INK, va="bottom", ha="left")
    return txt

def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)

def year_axis(ax, every=1, start=2019, end=2025, short=True):
    yrs = list(range(start, end + 1, every))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in yrs])
    ax.set_xticklabels([f"'{str(y)[2:]}" if short else str(y) for y in yrs])
    ax.set_xlim(MDATE[0], MDATE[-1])

# ---------------------------------------------------------------- data --------------------------------
def rd(name):
    return pd.read_parquet(os.path.join(DATA, f"agg_{name}.parquet"))

def tab(name):
    return pd.read_csv(os.path.join(DATA, name))

_cache = {}
def base_tables():
    if "sk" not in _cache:
        sk = rd("skills"); tt = rd("totals").rename(columns={"n_post": "den"})
        prog = sk[sk["skill"].isin(PROG_SKILLS)].copy()
        prog["basket"] = prog.groupby(["month", "occupation4d"])["n_post"].transform("sum")
        prog["rel"] = prog["n_post"] / prog["basket"] * 100
        prog = prog.merge(tt, on=["month", "occupation4d"])
        prog["post"] = prog["n_post"] / prog["den"] * 100
        _cache.update(sk=sk, tt=tt, prog=prog)
    return _cache["sk"], _cache["tt"], _cache["prog"]

def series(skill, occ, col="rel"):
    _, _, prog = base_tables()
    x = prog[(prog["skill"] == skill) & (prog["occupation4d"] == occ)].set_index("month")
    return x[col].reindex(MLIST, fill_value=0.0).to_numpy(float), x["n_post"].reindex(MLIST, fill_value=0.0).to_numpy(float)

# ---------------------------------------------------------------- the stage rule ----------------------
FLOOR_COUNT, MIN_MONTHS = 5, 36
BASE   = dict(W=12, theta=0.80, gamma=0.10, delta=0.10, rho=0.20, k=3)
RANGES = dict(W=[9, 12, 15], theta=(0.70, 0.90), gamma=(0.05, 0.15), delta=(0.05, 0.20), rho=(0.10, 0.30), k=[2, 3, 4])
N_DRAWS = 2000
STAGES = ["Not yet observed", "Birth", "Growth", "Maturity", "Decline", "Revival", "Death"]
NYO, BIRTH, GROWTH, MATURITY, DECLINE, REVIVAL, DEATH = range(7)

def roll(x, W):
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out[W - 1:] = (cs[W:] - cs[:-W]) / W
    return out

def level(x, W):
    return pd.Series(x).rolling(W).median().to_numpy()

def classify(s, c, W, theta, gamma, delta, rho, k, timeline=False, detail=False):
    n = len(s)
    c3 = roll(c, 3)
    born = np.where(c3 >= FLOOR_COUNT)[0]
    st = np.full(n, NYO)
    Pt = np.full(n, np.nan)
    if len(born) == 0:
        return (st, Pt) if detail else (st if timeline else NYO)
    b = born[0] - 2
    censored = b <= 0
    L, Lc = level(s, W), roll(c, W)
    first = max(W - 1 + 12, b + 12)
    if not censored:
        st[b:min(b + 12, n)] = BIRTH
    P, trough, cur, cand, run = 0.0, np.inf, -1, -1, 0
    for t in range(max(W - 1, b), n):
        if L[t] > P:
            P, trough = L[t], L[t]
        trough = min(trough, L[t])
        Pt[t] = P
        if t < first:
            continue
        g = L[t] / L[t - 12] - 1 if L[t - 12] > 0 else (np.inf if L[t] > 0 else 0.0)
        r = L[t] / P if P > 0 else 0.0
        if r < delta or Lc[t] < FLOOR_COUNT:
            raw = DEATH
        elif r >= theta:
            raw = GROWTH if g >= gamma else MATURITY
        else:
            raw = REVIVAL if (g >= gamma and L[t] >= (1 + rho) * trough) else DECLINE
        if cur == -1:
            cur = raw
        elif raw != cur:
            run = run + 1 if raw == cand else 1
            cand = raw
            if run >= k:
                cur, run = raw, 0
        else:
            run = 0
        st[t] = cur
    if detail:
        return st, Pt
    return st if timeline else int(st[-1])

def all_pairs():
    """{(skill, occ): dict(s, c, L, stages, P, observed)} on the relative share, baseline parameters."""
    if "pairs" in _cache:
        return _cache["pairs"]
    out = {}
    for skl in PROG_SKILLS:
        for occ in OCCUPATIONS:
            s, c = series(skl, occ)
            sp, _ = series(skl, occ, "post")
            observed = int((c > 0).sum()) >= MIN_MONTHS
            d = dict(s=s, c=c, post=sp, observed=observed)
            if observed:
                st, P = classify(s, c, **BASE, detail=True)
                d.update(L=level(s, BASE["W"]), stages=st, P=P, stage=STAGES[int(st[-1])])
            out[(skl, occ)] = d
    _cache["pairs"] = out
    return out

def stage_probabilities():
    """Stage shares over the joint draws; identical seed and draw order as the analysis notebook."""
    if "prob" in _cache:
        return _cache["prob"]
    rng = np.random.default_rng(2026)
    draws = pd.DataFrame({
        "W": rng.choice(RANGES["W"], N_DRAWS), "theta": rng.uniform(*RANGES["theta"], N_DRAWS),
        "gamma": rng.uniform(*RANGES["gamma"], N_DRAWS), "delta": rng.uniform(*RANGES["delta"], N_DRAWS),
        "rho": rng.uniform(*RANGES["rho"], N_DRAWS), "k": rng.choice(RANGES["k"], N_DRAWS)})
    rows = {}
    for key, d in all_pairs().items():
        if not d["observed"]:
            continue
        end = np.array([classify(d["s"], d["c"], int(p.W), p.theta, p.gamma, p.delta, p.rho, int(p.k))
                        for p in draws.itertuples(index=False)])
        rows[key] = {STAGES[x]: float((end == x).mean()) for x in range(len(STAGES))}
    prob = pd.DataFrame(rows).T
    prob.index.names = ["skill", "occupation"]
    _cache["prob"] = prob
    return prob

# ---------------------------------------------------------------- shared ordering and small helpers ---
def skill_order():
    """Skills from declining to growing: mean over occupations of P(Growth) - P(Decline) - P(Death)."""
    prob = stage_probabilities()
    sc = (prob["Growth"] + prob.get("Revival", 0) - prob["Decline"] - prob.get("Death", 0)).groupby(level=0).mean()
    return list(sc.sort_values().index)

def runs(st):
    """[(start, end_exclusive, stage_code)] for a stage timeline."""
    out, a = [], 0
    for i in range(1, len(st) + 1):
        if i == len(st) or st[i] != st[a]:
            out.append((a, i, int(st[a]))); a = i
    return out

def stage_legend(fig, x, y, stages=("Growth", "Maturity", "Decline", "Revival"), extra=None, fontsize=7, dx=0.105, anchor="left"):
    """A row of colour chips with ink labels, in figure coordinates."""
    from matplotlib.patches import Rectangle
    items = list(stages) + ([extra] if extra else [])
    w, h = 0.016, 0.012
    for i, s in enumerate(items):
        xi = x + i * dx
        if s == "Not observed":
            fig.patches.append(Rectangle((xi, y), w, h, transform=fig.transFigure, facecolor="white", edgecolor=AXIS, hatch="////", lw=0.5))
        else:
            fig.patches.append(Rectangle((xi, y), w, h, transform=fig.transFigure, facecolor=STAGE_COL[s], edgecolor="none"))
        fig.text(xi + w + 0.006, y + h / 2, s, fontsize=fontsize, color=INK, va="center", ha="left")
