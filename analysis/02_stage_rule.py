# ===== STEP 2: one stage rule, per pair, with stage probabilities =====
# Paste this whole file into ONE new cell at the bottom of the same notebook and run it.
#
# Reads the Step 0 output in `OUT_DIR` (no raw files needed, runs in about a minute).
#
# **One algorithm** assigns a stage to every skill–occupation pair in every month, from observed data only:
#
# * `L` = trailing `W`-month median of the posting share, so a posting burst shorter than half the window cannot
#   create a peak; `P` = highest `L` so far (running peak); `r = L / P`; `g` = change of `L` against 12 months earlier.
# * **Birth**: first 12 months after a skill first appears, when that happens inside the window.
#   A pair already present in the first month is *left-censored*: its Birth is not observed.
# * **Death**: `r < delta`, or fewer than `FLOOR_COUNT` postings a month. This means near-absence, not decline.
# * **Growth**: `r >= theta` and `g >= gamma`.  **Maturity**: `r >= theta` and `g < gamma`.
# * **Revival**: `r < theta`, `g >= gamma`, and `L` at least `rho` above its lowest value since the peak.
# * **Decline**: `r < theta` otherwise.
# * A stage changes only after `k` consecutive months. Growth can follow Maturity and Revival can follow Decline,
#   so the rule is not limited to a single peak.
#
# **Joint sensitivity**: all six parameters are drawn together `N_DRAWS` times from the ranges below and every
# pair is re-classified each time, which gives a probability for each stage instead of a single label.

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT_DIR     = os.environ.get("SMX_OUT_DIR", "/content/smx_r1")
START_MONTH = os.environ.get("SMX_START", "2018-10")
END_MONTH   = "2025-03"

PROG_SKILLS = [
    "computer programming", "java (computer programming)", "sql", "php",
    "web programming", "use object-oriented programming",
    "use scripting programming", "python (computer programming)", "css",
    "typescript", "object-oriented modelling", "ict system programming", "abap",
]
OCCUPATIONS = ["Software developers", "Systems analysts", "Web and multimedia developers", "Applications programmers"]

FLOOR_COUNT = 5        # fewer postings a month than this counts as absent
MIN_MONTHS  = 36       # a pair needs at least this many months with postings to be classified

BASE   = dict(W=12, theta=0.80, gamma=0.10, delta=0.10, rho=0.20, k=3)
RANGES = dict(W=[9, 12, 15], theta=(0.70, 0.90), gamma=(0.05, 0.15), delta=(0.05, 0.20), rho=(0.10, 0.30), k=[2, 3, 4])
N_DRAWS = int(os.environ.get("SMX_NDRAWS", "2000"))
ROBUST  = 0.85         # a stage held in at least this share of draws is reported as robust
STAGES  = ["Not yet observed", "Birth", "Growth", "Maturity", "Decline", "Revival", "Death"]
NYO, BIRTH, GROWTH, MATURITY, DECLINE, REVIVAL, DEATH = range(7)

MONTHS = pd.period_range(START_MONTH, END_MONTH, freq="M")
sk = pd.read_parquet(os.path.join(OUT_DIR, "agg_skills.parquet"))
tt = pd.read_parquet(os.path.join(OUT_DIR, "agg_totals.parquet")).rename(columns={"n_post": "den"})
sd = sk[sk["skill"].isin(PROG_SKILLS)].merge(tt, on=["month", "occupation4d"])
sd["share"] = sd["n_post"] / sd["den"] * 100

def series(skill, occ):
    x = sd[(sd["skill"] == skill) & (sd["occupation4d"] == occ)]
    idx = pd.PeriodIndex(x["month"], freq="M")
    s = pd.Series(x["share"].to_numpy(float), index=idx).reindex(MONTHS, fill_value=0.0)
    c = pd.Series(x["n_post"].to_numpy(float), index=idx).reindex(MONTHS, fill_value=0.0)
    return s.to_numpy(), c.to_numpy()

def roll(x, W):
    """Trailing W-month mean."""
    out = np.full(len(x), np.nan)
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out[W - 1:] = (cs[W:] - cs[:-W]) / W
    return out

def level(x, W):
    """Trailing W-month median: a posting burst shorter than half the window cannot create a peak."""
    return pd.Series(x).rolling(W).median().to_numpy()

def classify(s, c, W, theta, gamma, delta, rho, k, timeline=False):
    """Stage of one pair in every month. Returns the last stage, or the whole timeline."""
    n = len(s)
    c3 = roll(c, 3)
    born = np.where(c3 >= FLOOR_COUNT)[0]
    st = np.full(n, NYO)
    if len(born) == 0:
        return st if timeline else NYO
    b = born[0] - 2                                   # first month of the first sustained appearance
    censored = b <= 0
    L, Lc = level(s, W), roll(c, W)
    first = max(W - 1 + 12, b + 12)                   # first month with a full window and a 12-month comparison
    if not censored:
        st[b:min(b + 12, n)] = BIRTH
    P, trough, cur, cand, run = 0.0, np.inf, -1, -1, 0
    for t in range(max(W - 1, b), n):
        if L[t] > P:
            P, trough = L[t], L[t]
        trough = min(trough, L[t])
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
    return st if timeline else int(st[-1])

# ---- baseline -----------------------------------------------------------------------------------
rows, tl = [], []
data = {}
for skl in PROG_SKILLS:
    for occ in OCCUPATIONS:
        s, c = series(skl, occ)
        observed = int((c > 0).sum())
        if observed < MIN_MONTHS:
            rows.append({"skill": skl, "occupation": occ, "months_observed": observed, "stage": "Not observed"})
            continue
        data[(skl, occ)] = (s, c)
        st = classify(s, c, **BASE, timeline=True)
        L = level(s, BASE["W"])
        born = np.where(roll(c, 3) >= FLOOR_COUNT)[0]
        b = born[0] - 2 if len(born) else None
        ip = int(np.nanargmax(L))
        rows.append({
            "skill": skl, "occupation": occ, "months_observed": observed,
            "birth": "before window (left-censored)" if (b is not None and b <= 0) else (str(MONTHS[b]) if b is not None else ""),
            "peak_month": str(MONTHS[ip]), "peak_share": round(float(L[ip]), 3), "current_share": round(float(L[-1]), 3),
            "current_vs_peak": round(float(L[-1] / L[ip]), 3) if L[ip] > 0 else np.nan,
            "change_12m_%": round(float((L[-1] / L[-13] - 1) * 100), 1) if L[-13] > 0 else np.nan,
            "postings_per_month_now": round(float(c[-12:].mean()), 0),
            "stage": STAGES[int(st[-1])],
            "stage_path": " > ".join(STAGES[x] for i, x in enumerate(st) if x != NYO and (i == 0 or st[i - 1] != x)),
        })
        tl.append(pd.DataFrame({"skill": skl, "occupation": occ, "month": MONTHS.astype(str),
                                "share": s, "stage": [STAGES[x] for x in st]}))
base = pd.DataFrame(rows)
pd.concat(tl).to_csv(os.path.join(OUT_DIR, "tab6_stage_timeline.csv"), index=False)

# ---- joint sensitivity: all parameters drawn together -------------------------------------------
rng = np.random.default_rng(2026)
draws = pd.DataFrame({
    "W":     rng.choice(RANGES["W"], N_DRAWS),
    "theta": rng.uniform(*RANGES["theta"], N_DRAWS),
    "gamma": rng.uniform(*RANGES["gamma"], N_DRAWS),
    "delta": rng.uniform(*RANGES["delta"], N_DRAWS),
    "rho":   rng.uniform(*RANGES["rho"], N_DRAWS),
    "k":     rng.choice(RANGES["k"], N_DRAWS),
})
pairs = list(data)
res = np.zeros((N_DRAWS, len(pairs)), dtype=int)
for j, key in enumerate(pairs):
    s, c = data[key]
    for i, p in enumerate(draws.itertuples(index=False)):
        res[i, j] = classify(s, c, int(p.W), p.theta, p.gamma, p.delta, p.rho, int(p.k))

prob = pd.DataFrame({STAGES[x]: (res == x).mean(axis=0) for x in range(len(STAGES))}, index=pd.MultiIndex.from_tuples(pairs, names=["skill", "occupation"]))
prob = prob.loc[:, prob.sum() > 0].round(3)
summ = pd.DataFrame({"modal_stage": prob.idxmax(axis=1), "p_modal": prob.max(axis=1)})
out = base.merge(summ.reset_index(), on=["skill", "occupation"], how="left").merge(prob.add_prefix("p_").reset_index(), on=["skill", "occupation"], how="left")
out["robust"] = np.where(out["p_modal"].isna(), "", np.where(out["p_modal"] >= ROBUST, "yes", "no"))
out.to_csv(os.path.join(OUT_DIR, "tab5_stages_by_pair.csv"), index=False)

# which parameter moves the labels: agreement with the baseline label, by value of each parameter
base_lab = np.array([STAGES.index(base.set_index(["skill", "occupation"]).loc[k, "stage"]) for k in pairs])
agree = (res == base_lab).mean(axis=1)
imp = []
for p in draws.columns:
    grp = draws[p] if draws[p].nunique() <= 5 else pd.qcut(draws[p], 5)
    a = pd.Series(agree).groupby(grp.to_numpy(), observed=True).mean()
    imp.append({"parameter": p, "baseline": BASE[p], "range": str(RANGES[p]),
                "agreement_lowest_%": round(a.min() * 100, 1), "agreement_highest_%": round(a.max() * 100, 1),
                "spread_pp": round((a.max() - a.min()) * 100, 1)})
imp = pd.DataFrame(imp).sort_values("spread_pp", ascending=False)
imp.to_csv(os.path.join(OUT_DIR, "tab7_sensitivity_parameters.csv"), index=False)

# ---- print ------------------------------------------------------------------------------------------
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
short = {"Software developers": "SW dev", "Systems analysts": "Sys analysts",
         "Web and multimedia developers": "Web dev", "Applications programmers": "App prog"}
v = out.copy(); v["occupation"] = v["occupation"].map(short)
print(f"Window {MONTHS[0]} to {MONTHS[-1]} ({len(MONTHS)} months). Classified pairs: {len(pairs)} of {len(base)}.")
print(f"Baseline parameters: {BASE}\n")
cols = ["skill", "occupation", "birth", "peak_month", "peak_share", "current_share", "current_vs_peak", "change_12m_%",
        "stage", "modal_stage", "p_modal", "robust"]
print(v[cols].to_string(index=False))
print("\nBaseline stage, number of pairs:")
print(out["stage"].value_counts().to_string())
print(f"\nPairs whose stage holds in at least {ROBUST:.0%} of {N_DRAWS} joint draws: "
      f"{(out['robust'] == 'yes').sum()} of {len(pairs)}")
print(f"Agreement with the baseline label across draws: mean {agree.mean():.1%}, "
      f"5th percentile {np.percentile(agree, 5):.1%}, minimum {agree.min():.1%}")
print("\nWhich parameter moves the labels (agreement with baseline by parameter value):")
print(imp.to_string(index=False))
print("\nStage paths over time (baseline):")
print(v.loc[v["stage_path"].fillna("") != "", ["skill", "occupation", "stage_path"]].to_string(index=False))

sel_path = os.path.join(OUT_DIR, "tab_selected_models_raw.csv")
if os.path.exists(sel_path):
    pk = pd.read_csv(sel_path)
    pk = pk[pk["origin"] <= "2023-12"]
    print("\nStep 1, model picked on the validation year (share of pair-origins):")
    print((pk["picked"].value_counts(normalize=True) * 100).round(1).to_string())
print(f"\nFiles written to {OUT_DIR}: tab5_stages_by_pair.csv, tab6_stage_timeline.csv, tab7_sensitivity_parameters.csv")
