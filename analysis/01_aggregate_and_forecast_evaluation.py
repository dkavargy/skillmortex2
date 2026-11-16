# %% [markdown]
# # SkillMORTEX 2.0: aggregation and forecast evaluation
#
# **Step 0** builds the monthly series from the raw online job advertisement files (one row per advertisement and skill)
# and audits them: files, rows, distinct advertisements, rows per advertisement.
# **Step 1** evaluates every candidate model against the naive and the seasonal naive forecast for all skill-occupation
# pairs, at horizons of 1-12, 13-24 and 25-36 months, with coverage of the 80% prediction interval.
#
# `MODE` (environment variable `SMX_MODE`):
#
# * `relative` (the paper): share = advertisements that mention the skill / sum of that count over the 13 programming skills.
# * `raw`: share = advertisements that mention the skill / advertisements.
# * `mentions`: the share used in the first submission, from a pre-aggregated file.
#
# The raw files are not distributed. `SMX_DRIVE_FOLDER` names the Google Drive folder that holds them when the script
# runs in Colab; otherwise place the CSV files in `RAW_DIR`. Everything is written to `OUT_DIR`.
# %%
import os, glob, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODE      = os.environ.get("SMX_MODE", "relative")       # "relative" (paper), "raw" (share of ads) or "mentions" (first submission)
RAW_DIR   = os.environ.get("SMX_RAW_DIR", "/content/software-occupations")
OLD_PARQ  = os.environ.get("SMX_OLD_PARQ", "/content/sample_data/monthly_skills_agg (2).parquet")
OUT_DIR   = os.environ.get("SMX_OUT_DIR", "/content/smx_r1")
os.makedirs(OUT_DIR, exist_ok=True)

START_MONTH = os.environ.get("SMX_START", "2018-07")     # set to "2018-10" to drop the first three months
END_MONTH   = "2025-03"

PROG_SKILLS = [
    "computer programming", "java (computer programming)", "sql", "php",
    "web programming", "use object-oriented programming",
    "use scripting programming", "python (computer programming)", "css",
    "typescript", "object-oriented modelling", "ict system programming", "abap",
]
OCCUPATIONS = [
    "Software developers", "Systems analysts",
    "Web and multimedia developers", "Applications programmers",
]

# A pair enters the evaluation only if the skill is observed in at least MIN_NONZERO months
# and is mentioned by at least MIN_MEAN_COUNT postings per month on average.
MIN_NONZERO    = 72
MIN_MEAN_COUNT = 30

# Forecast origins (last month of the training window). Each origin forecasts up to 36 months ahead.
ORIGINS = ["2020-12", "2021-06", "2021-12", "2022-06", "2022-12", "2023-06", "2023-12"]
ORIGINS = os.environ.get("SMX_ORIGINS", ",".join(ORIGINS)).split(",")
MAX_H   = 36
BUCKETS = {"h01-12": (1, 12), "h13-24": (13, 24), "h25-36": (25, 36)}
USE_PROPHET = os.environ.get("SMX_PROPHET", "1") == "1"
N_SIM   = int(os.environ.get("SMX_NSIM", "200"))          # bootstrap paths for Holt-Winters intervals
Z80     = 1.2816

# %% [markdown]
# ## Raw files from Google Drive
#
# Same logic as cells 0 and 1 of the original notebook: authenticate, list the CSV files in the Drive folder,
# download the ones that are not already in `RAW_DIR`. Skipped when Step 0 output already exists.

# %%
FOLDER_ID = os.environ.get("SMX_DRIVE_FOLDER", "")   # Google Drive folder that holds the raw CSV export (not distributed)

def download_from_drive(folder_id, dest):
    import re
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    os.makedirs(dest, exist_ok=True)
    stack, files, seen = [folder_id], [], set()
    while stack:
        fid = stack.pop()
        if fid in seen:
            continue
        seen.add(fid)
        for item in drive.ListFile({"q": f"'{fid}' in parents and trashed=false"}).GetList():
            if item["mimeType"] == "application/vnd.google-apps.folder":
                stack.append(item["id"])
            elif re.search(r"\.csv$", item["title"], re.IGNORECASE):
                files.append(item)
    print(f"{len(files)} CSV files in the Drive folder")
    missing, redone = [], []
    for i, f in enumerate(files, 1):
        safe = f"{f['id']}__" + re.sub(r'[\\/:*?"<>|]+', "_", f["title"])
        path = os.path.join(dest, safe)
        want = int(f["fileSize"]) if f.get("fileSize") else None          # size of the file on Drive, in bytes
        def complete():
            return os.path.exists(path) and os.path.getsize(path) > 0 and (want is None or os.path.getsize(path) == want)
        if complete():
            continue
        if os.path.exists(path):
            print(f"  [{i}/{len(files)}] {f['title']}: local copy has {os.path.getsize(path):,} bytes, "
                  f"Drive has {want:,}. Downloading again.")
            os.remove(path); redone.append(f["title"])
        for attempt in range(1, 7):                      # Drive allows a limited number of requests per minute
            try:
                f.GetContentFile(path)
                if not complete():
                    raise IOError("incomplete download")
                break
            except Exception as e:
                if os.path.exists(path):
                    os.remove(path)                       # never keep a partial file
                wait = min(15 * 2 ** (attempt - 1), 90)
                print(f"  [{i}/{len(files)}] {f['title']}: attempt {attempt} failed, waiting {wait}s")
                time.sleep(wait)
        else:
            missing.append(f["title"])
    if redone:
        print(f"Replaced {len(redone)} incomplete files: {redone}")
    n_ok = len(files) - len(missing)
    print(f"{n_ok} of {len(files)} files ready in {dest}")
    if missing:
        raise RuntimeError(f"{len(missing)} files could not be downloaded: {missing}. "
                           "Run this cell again in a few minutes; files already downloaded are kept.")
    return len(files)

import json
MANIFEST = os.path.join(OUT_DIR, "step0_manifest.json")
manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
N_DRIVE = manifest.get("n_drive")

if MODE in ("raw", "relative"):
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except ImportError:
        in_colab = False
        print(f"Not running in Colab; using the CSV files already in {RAW_DIR}")
    def local_state():
        ps = glob.glob(os.path.join(RAW_DIR, "*.csv"))
        return len(ps), int(sum(os.path.getsize(q) for q in ps))
    n_local, bytes_local = local_state()
    verified = bool(manifest) and manifest.get("n_files") == n_local == N_DRIVE and manifest.get("bytes") == bytes_local
    if in_colab and not verified:
        N_DRIVE = download_from_drive(FOLDER_ID, RAW_DIR)      # every file is checked against its size on Drive

# %% [markdown]
# ## Step 0 — posting-level aggregation and file audit
#
# The submitted paper divides the number of *skill mentions* of skill k by the number of *all skill mentions*
# in the occupation and month. Here the unit is the posting: distinct `oja_id` that mention skill k,
# divided by distinct `oja_id` in the occupation and month.

# %%
RAW_COLS = ["oja_id", "first_active_date", "skill", "occupation4d", "country", "source_stability"]

def step0_aggregate(raw_dir):
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {raw_dir}. Run the download cells first.")
    try:
        from tqdm import tqdm
        it = tqdm(paths, desc="Step 0")
    except Exception:
        it = paths

    audit, id_chunks = [], []
    tot, num, tot_c, num_c, tot_s, num_s = [], [], [], [], [], []
    prog = set(PROG_SKILLS)

    for p in it:
        df = pd.read_csv(p, usecols=lambda c: c in RAW_COLS, low_memory=False, on_bad_lines="skip")
        n_rows = len(df)
        df["month"] = pd.to_datetime(df["first_active_date"], errors="coerce").dt.strftime("%Y-%m")
        df = df.dropna(subset=["month", "oja_id", "occupation4d"])
        df["skill"] = df["skill"].astype("string").str.lower().str.strip()
        for c in ("country", "source_stability"):
            if c not in df.columns:
                df[c] = "NA"
            df[c] = df[c].fillna("NA").astype(str)

        ids = df["oja_id"].drop_duplicates()
        id_chunks.append(pd.util.hash_pandas_object(ids, index=False).to_numpy())
        audit.append({"file": os.path.basename(p), "rows": n_rows, "postings": len(ids),
                      "rows_no_skill": int(df["skill"].isna().sum())})

        k = ["month", "occupation4d"]
        tot.append(df.groupby(k)["oja_id"].nunique().rename("n_post").reset_index())
        tot_c.append(df.groupby(k + ["country"])["oja_id"].nunique().rename("n_post").reset_index())
        tot_s.append(df.groupby(k + ["source_stability"])["oja_id"].nunique().rename("n_post").reset_index())

        ds = df.dropna(subset=["skill"])
        g = ds.groupby(k + ["skill"])["oja_id"].agg(n_post="nunique", n_mentions="size").reset_index()
        num.append(g)
        dp = ds[ds["skill"].isin(prog)]
        num_c.append(dp.groupby(k + ["country", "skill"])["oja_id"].nunique().rename("n_post").reset_index())
        num_s.append(dp.groupby(k + ["source_stability", "skill"])["oja_id"].nunique().rename("n_post").reset_index())

    def comb(frames, keys):
        return pd.concat(frames, ignore_index=True).groupby(keys, as_index=False).sum(numeric_only=True)

    k = ["month", "occupation4d"]
    out = {
        "totals":        comb(tot, k),
        "skills":        comb(num, k + ["skill"]),
        "totals_country": comb(tot_c, k + ["country"]),
        "skills_country": comb(num_c, k + ["country", "skill"]),
        "totals_stab":   comb(tot_s, k + ["source_stability"]),
        "skills_stab":   comb(num_s, k + ["source_stability", "skill"]),
    }
    for name, d in out.items():
        d.to_parquet(os.path.join(OUT_DIR, f"agg_{name}.parquet"), index=False)

    # ---- audit ---------------------------------------------------------------------------------
    audit = pd.DataFrame(audit)
    audit.to_csv(os.path.join(OUT_DIR, "audit_files.csv"), index=False)
    all_ids = np.concatenate(id_chunks)
    n_sum, n_unique = int(audit["postings"].sum()), int(len(np.unique(all_ids)))

    lines = []
    lines.append(f"Files read: {len(audit)}   rows: {int(audit['rows'].sum()):,}")
    lines.append(f"Distinct postings, summed over files: {n_sum:,}")
    lines.append(f"Distinct postings, whole corpus:      {n_unique:,}   (paper reports 14,157,581)")
    if n_sum != n_unique:
        lines.append(f"WARNING: {n_sum - n_unique:,} postings appear in more than one file. "
                     "Per-file sums overcount them; tell Claude before using the shares.")
    top = audit["rows"].value_counts()
    if top.iloc[0] >= 3 and top.index[0] == audit["rows"].max():
        lines.append(f"WARNING: {top.iloc[0]} files have exactly {top.index[0]:,} rows, the maximum. "
                     "The export may be capped at that number of rows.")
    by_occ = out["totals"].groupby("occupation4d")["n_post"].sum().sort_values(ascending=False)
    lines.append("\nPostings per occupation (compare with Table 2 of the paper):")
    lines += [f"  {o:32s} {int(v):>12,}  {v / by_occ.sum() * 100:5.2f}%" for o, v in by_occ.items()]

    grid = out["totals"].pivot_table(index="month", columns="occupation4d", values="n_post", aggfunc="sum")
    grid = grid.reindex(pd.period_range(grid.index.min(), grid.index.max(), freq="M").astype(str))
    gaps = [(o, mth) for o in grid.columns for mth in grid.index[grid[o].isna()]]
    if gaps:
        lines.append(f"\nWARNING: {len(gaps)} occupation-months have no postings at all, files are probably missing: "
                     + "; ".join(f"{o} {mth}" for o, mth in gaps[:12]) + (" ..." if len(gaps) > 12 else ""))
    else:
        lines.append(f"\nEvery occupation has postings in every month from {grid.index.min()} to {grid.index.max()}.")

    m = out["totals"].groupby("month")["n_post"].sum().rename("postings").to_frame()
    m["mentions"] = out["skills"].groupby("month")["n_mentions"].sum()
    m["skills_per_posting"] = (m["mentions"] / m["postings"]).round(2)
    m.to_csv(os.path.join(OUT_DIR, "audit_monthly_volume.csv"))
    med = m["postings"].median()
    odd = m[(m["postings"] > 1.6 * med) | (m["postings"] < 0.5 * med)]
    lines.append("\nFirst six months (postings, skill mentions, skills per posting):")
    lines.append(m.head(6).to_string())
    lines.append(f"\nMedian monthly postings: {med:,.0f}. Months above 1.6x or below 0.5x the median: "
                 f"{', '.join(odd.index) if len(odd) else 'none'}")
    lines.append(f"Skills per posting: min {m['skills_per_posting'].min()} ({m['skills_per_posting'].idxmin()}), "
                 f"max {m['skills_per_posting'].max()} ({m['skills_per_posting'].idxmax()})")
    txt = "\n".join(lines)
    open(os.path.join(OUT_DIR, "audit_summary.txt"), "w").write(txt)
    print(txt)
    if gaps and os.environ.get("SMX_ALLOW_GAPS") != "1":
        raise RuntimeError("Stopping: some occupation-months have no data (see the warning). "
                           "Run the Drive cell again to fetch the missing files, then rerun this cell.")
    return out

if MODE in ("raw", "relative"):
    n_local, bytes_local = local_state()
    fresh = (os.path.exists(os.path.join(OUT_DIR, "agg_skills.parquet")) and manifest.get("n_files") == n_local
             and manifest.get("bytes") == bytes_local
             and os.environ.get("SMX_FORCE0") != "1")
    if fresh:
        print(f"Step 0 output found (built from {n_local} files), reusing it.")
        print(open(os.path.join(OUT_DIR, "audit_summary.txt")).read())
    else:
        _ = step0_aggregate(RAW_DIR)
        manifest = {"n_files": n_local, "n_drive": N_DRIVE, "bytes": bytes_local}
        json.dump(manifest, open(MANIFEST, "w"))
        stale = os.path.join(OUT_DIR, "eval_long_raw.parquet")       # forecasts made from an older aggregation
        if os.path.exists(stale):
            os.remove(stale)

# %% [markdown]
# ## Monthly share series for the 13 skills x 4 occupations

# %%
MONTHS = pd.period_range(START_MONTH, END_MONTH, freq="M")

def load_counts():
    """Returns long frame: month, occupation4d, skill, num, den  (share = num / den * 100)."""
    if MODE in ("raw", "relative"):
        sk  = pd.read_parquet(os.path.join(OUT_DIR, "agg_skills.parquet"))
        tt  = pd.read_parquet(os.path.join(OUT_DIR, "agg_totals.parquet"))
        num = sk[sk["skill"].isin(PROG_SKILLS)].rename(columns={"n_post": "num"})
        den = tt.rename(columns={"n_post": "den"})
        old = sk.groupby(["month", "occupation4d"])["n_mentions"].sum().rename("den_mentions").reset_index()
        d = num.merge(den, on=["month", "occupation4d"]).merge(old, on=["month", "occupation4d"])
        d["share_mentions"] = d["n_mentions"] / d["den_mentions"] * 100
    else:
        mo = pd.read_parquet(OLD_PARQ)
        mo["month"] = mo["month"].astype(str)
        mo["skill"] = mo["skill"].str.lower().str.strip()
        den = mo.groupby(["month", "occupation4d"])["postings"].sum().rename("den").reset_index()
        num = (mo[mo["skill"].isin(PROG_SKILLS)]
               .groupby(["month", "occupation4d", "skill"])["postings"].sum().rename("num").reset_index())
        d = num.merge(den, on=["month", "occupation4d"])
    if MODE == "relative":      # the measure used in the paper: the skill's part of the 13 programming skills, per occupation and month
        d["den"] = d.groupby(["month", "occupation4d"])["num"].transform("sum")
    d["share"] = d["num"] / d["den"] * 100
    return d

counts = load_counts()

def get_series(skill, occ, col="share"):
    d = counts[(counts["skill"] == skill) & (counts["occupation4d"] == occ)]
    s = pd.Series(d[col].to_numpy(dtype=float), index=pd.PeriodIndex(d["month"], freq="M"))
    return s.reindex(MONTHS, fill_value=0.0)

rows = []
for sk in PROG_SKILLS:
    for oc in OCCUPATIONS:
        s, c = get_series(sk, oc), get_series(sk, oc, "num")
        r = {"skill": sk, "occupation": oc, "months_nonzero": int((c > 0).sum()),
             "mean_monthly_count": round(float(c.mean()), 1), "mean_share": round(float(s.mean()), 3),
             "first_month": str(c[c > 0].index.min()) if (c > 0).any() else "",
             "share_first12": round(float(s.iloc[:12].mean()), 3), "share_last12": round(float(s.iloc[-12:].mean()), 3)}
        if MODE in ("raw", "relative"):
            sm = get_series(sk, oc, "share_mentions")
            r["mean_share_mentions_old"] = round(float(sm.mean()), 3)
            r["corr_new_vs_old"] = round(float(np.corrcoef(s, sm)[0, 1]), 3) if s.std() > 0 and sm.std() > 0 else np.nan
            r["last12_vs_first12_new_%"] = round((s.iloc[-12:].mean() / max(s.iloc[:12].mean(), 1e-9) - 1) * 100, 1)
            r["last12_vs_first12_old_%"] = round((sm.iloc[-12:].mean() / max(sm.iloc[:12].mean(), 1e-9) - 1) * 100, 1)
        r["eligible"] = bool(r["months_nonzero"] >= MIN_NONZERO and r["mean_monthly_count"] >= MIN_MEAN_COUNT)
        rows.append(r)
coverage = pd.DataFrame(rows)
coverage.to_csv(os.path.join(OUT_DIR, "tab0_pair_coverage.csv"), index=False)
PAIRS = [(r.skill, r.occupation) for r in coverage.itertuples() if r.eligible]
print(f"Share definition: {'ads mentioning the skill / sum of that count over the 13 skills' if MODE == 'relative' else 'ads mentioning the skill / ads' if MODE == 'raw' else 'skill mentions / all skill mentions (first submission)'}")
print(f"Pairs with enough data: {len(PAIRS)} of {len(coverage)}")
print("Excluded pairs:")
print(coverage.loc[~coverage["eligible"], ["skill", "occupation", "months_nonzero", "mean_monthly_count"]].to_string(index=False))

# %% [markdown]
# ## Step 1 — models, benchmarks and expanding-window evaluation
#
# A model that fails to fit is recorded as failed. It is never replaced by another model's forecast.

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

if USE_PROPHET:
    try:
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        from prophet import Prophet
    except Exception as e:
        print("Prophet not available:", e); USE_PROPHET = False

BENCHMARKS = ["Naive", "SeasonalNaive"]

def f_naive(y, idx, H):
    sd = np.std(np.diff(y), ddof=1)
    m  = np.repeat(y[-1], H)
    hw = Z80 * sd * np.sqrt(np.arange(1, H + 1))
    return m, m - hw, m + hw

def f_snaive(y, idx, H):
    m  = np.array([y[len(y) - 12 + (i % 12)] for i in range(H)])      # same month, most recent year
    sd = np.std(y[12:] - y[:-12], ddof=1)
    hw = Z80 * sd * np.sqrt(np.arange(H) // 12 + 1)
    return m, m - hw, m + hw

def _sarima(order, sorder):
    def f(y, idx, H):
        fit = SARIMAX(y, order=order, seasonal_order=sorder).fit(disp=False, maxiter=200)
        fc  = fit.get_forecast(H)
        ci  = np.asarray(fc.conf_int(alpha=0.20))
        return np.asarray(fc.predicted_mean), ci[:, 0], ci[:, 1]
    return f

def _hw(seasonal, damped):
    def f(y, idx, H):
        yy  = np.clip(y, 1e-6, None) if seasonal == "mul" else y
        fit = ExponentialSmoothing(yy, trend="add", seasonal=seasonal, seasonal_periods=12,
                                   damped_trend=damped).fit(optimized=True)
        m = np.asarray(fit.forecast(H))
        try:
            sim = np.asarray(fit.simulate(H, repetitions=N_SIM, random_errors="bootstrap"))
            lo, hi = np.nanpercentile(sim, 10, axis=1), np.nanpercentile(sim, 90, axis=1)
        except Exception:
            lo = hi = np.full(H, np.nan)
        return m, lo, hi
    return f

def f_prophet(y, idx, H):
    dfp = pd.DataFrame({"ds": idx.to_timestamp(), "y": y})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.80)
    m.fit(dfp)
    fc = m.predict(m.make_future_dataframe(periods=H, freq="MS")).tail(H)
    return fc["yhat"].to_numpy(), fc["yhat_lower"].to_numpy(), fc["yhat_upper"].to_numpy()

MODELS = {
    "Naive": f_naive,
    "SeasonalNaive": f_snaive,
    "SARIMA(1,1,1)(1,1,1)12": _sarima((1, 1, 1), (1, 1, 1, 12)),
    "SARIMA(0,1,1)(0,1,1)12": _sarima((0, 1, 1), (0, 1, 1, 12)),
    "HW_Multiplicative": _hw("mul", False),
    "HW_Additive_Damped": _hw("add", True),
}
if USE_PROPHET:
    MODELS["Prophet"] = f_prophet

def run_origin(y_full, origin):
    """Fit every model on data up to `origin`, forecast up to MAX_H months, return one row per model and step."""
    train = y_full[:origin]
    test  = y_full[origin + 1:][:MAX_H]
    y, H  = train.to_numpy(dtype=float), len(test)
    out   = []
    if len(y) < 26 or H == 0:
        return out
    scale_s = np.mean(np.abs(y[12:] - y[:-12]))
    scale_1 = np.mean(np.abs(np.diff(y)))
    for name, fn in MODELS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m, lo, hi = fn(y, train.index, H)
            m = np.asarray(m, dtype=float)
            if (not np.all(np.isfinite(m))) or np.max(np.abs(m)) > 10 * max(y.max(), 1e-6):
                raise ValueError("unstable forecast")
            status = "ok"
        except Exception as e:
            m = lo = hi = np.full(H, np.nan); status = f"failed: {type(e).__name__}"
        m = np.clip(m, 0, None)
        for i in range(H):
            out.append((str(origin), name, i + 1, str(test.index[i]), float(test.iloc[i]),
                        float(m[i]), float(lo[i]) if np.isfinite(lo[i]) else np.nan,
                        float(hi[i]) if np.isfinite(hi[i]) else np.nan, scale_s, scale_1, len(y), status))
    return out

COLS = ["origin", "model", "h", "month", "y", "yhat", "lo", "hi", "scale_s", "scale_1", "n_train", "status"]
ck_path = os.path.join(OUT_DIR, f"eval_long_{MODE}.parquet")
done = pd.read_parquet(ck_path) if os.path.exists(ck_path) and os.environ.get("SMX_FORCE1") != "1" else None
frames = [done] if done is not None else []
done_pairs = set(zip(done["skill"], done["occupation"])) if done is not None else set()

t0 = time.time()
for n, (sk, oc) in enumerate(PAIRS, 1):
    if (sk, oc) in done_pairs:
        continue
    s = get_series(sk, oc)
    rec = []
    for o in ORIGINS:
        rec += run_origin(s, pd.Period(o, freq="M"))
    d = pd.DataFrame(rec, columns=COLS)
    d.insert(0, "occupation", oc); d.insert(0, "skill", sk)
    frames.append(d)
    pd.concat(frames, ignore_index=True).to_parquet(ck_path, index=False)      # checkpoint after every pair
    print(f"[{n}/{len(PAIRS)}] {sk} | {oc}   elapsed {time.time() - t0:,.0f}s")

ev = pd.concat(frames, ignore_index=True)
ev = ev[ev[["skill", "occupation"]].apply(tuple, axis=1).isin(set(PAIRS)) & ev["origin"].isin(ORIGINS)].copy()
ev["ae"] = (ev["y"] - ev["yhat"]).abs()
ev["se"] = (ev["y"] - ev["yhat"]) ** 2
ev["covered"] = np.where(ev["lo"].notna() & ev["hi"].notna(), ((ev["y"] >= ev["lo"]) & (ev["y"] <= ev["hi"])).astype(float), np.nan)
ev["bucket"] = pd.cut(ev["h"], [0, 12, 24, 36], labels=list(BUCKETS))
fails = ev[ev["status"] != "ok"].drop_duplicates(["skill", "occupation", "origin", "model"])
print(f"\nFits: {ev.drop_duplicates(['skill','occupation','origin','model']).shape[0]}   failed: {len(fails)}")
if len(fails):
    print(fails.groupby(["model", "status"]).size().to_string())

# %% [markdown]
# ### Model selection without looking at the test window
#
# For an origin *o*, the selected model is the one with the lowest MAE over the 12 months that end at *o*,
# forecast from origin *o* − 12. Benchmarks compete on equal terms. The first origins have no earlier
# validation window, so "Selected" starts at the third origin.

# %%
P = ["skill", "occupation"]
val = (ev[(ev["h"] <= 12) & (ev["status"] == "ok")]
       .groupby(P + ["origin", "model"], observed=True)["ae"].agg(["mean", "size"]).reset_index())
val = val[val["size"] == 12]
val["next_origin"] = (pd.PeriodIndex(val["origin"], freq="M") + 12).astype(str)
pick = (val.sort_values(P + ["next_origin", "mean", "model"]).groupby(P + ["next_origin"], as_index=False).first()
        [P + ["next_origin", "model"]].rename(columns={"next_origin": "origin", "model": "picked"}))
sel = ev.merge(pick, on=P + ["origin"])
sel = sel[sel["model"] == sel["picked"]].copy()
sel["model"] = "Selected"
ev2 = pd.concat([ev, sel.drop(columns="picked")], ignore_index=True)
pick.to_csv(os.path.join(OUT_DIR, f"tab_selected_models_{MODE}.csv"), index=False)

# %% [markdown]
# ### Tables

# %%
from scipy.stats import wilcoxon

def gmean(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else np.nan

def compare(df, label):
    """Pair-level MAE per model, then each model against both benchmarks on exactly the same cells."""
    out = []
    for b, g in df.groupby("bucket", observed=True):
        wide = g.pivot_table(index=P + ["origin", "h"], columns="model", values="ae", observed=True)
        cov  = g.groupby("model", observed=True)["covered"].mean()
        for m in wide.columns:
            row = {"window": label, "horizon": b, "model": m}
            for bench in BENCHMARKS:
                w = wide[[m, bench]].dropna()
                pm = w.groupby(level=[0, 1]).mean()                    # MAE per pair on common cells
                if m == bench or pm.empty:
                    continue
                rel = pm[m] / pm[bench].replace(0, np.nan)
                tag = "SN" if bench == "SeasonalNaive" else "N"
                row.update({f"pairs_vs_{tag}": int(rel.notna().sum()),
                            f"relMAE_vs_{tag}_gmean": round(gmean(rel), 3),
                            f"relMAE_vs_{tag}_median": round(float(rel.median()), 3),
                            f"pct_pairs_beating_{tag}": round(float((rel < 1).mean() * 100), 1)})
                try:
                    row[f"wilcoxon_p_vs_{tag}"] = round(float(wilcoxon(pm[m], pm[bench]).pvalue), 4) if len(pm) >= 6 else np.nan
                except Exception:
                    row[f"wilcoxon_p_vs_{tag}"] = np.nan
            pm_all = wide[m].dropna().groupby(level=[0, 1]).mean()
            row["pairs"] = int(len(pm_all)); row["MAE_own_cells"] = round(float(pm_all.mean()), 4)
            row["coverage80_%"] = round(float(cov.get(m, np.nan) * 100), 1)
            out.append(row)
    return pd.DataFrame(out)

tab1 = compare(ev2, "all origins")
front = ["window", "horizon", "model", "pairs", "relMAE_vs_SN_gmean", "pct_pairs_beating_SN",
         "wilcoxon_p_vs_SN", "relMAE_vs_N_gmean", "pct_pairs_beating_N", "wilcoxon_p_vs_N", "coverage80_%"]
tab1 = tab1[[c for c in front if c in tab1.columns] + [c for c in tab1.columns if c not in front]]
tab1.to_csv(os.path.join(OUT_DIR, f"tab1_models_by_horizon_{MODE}.csv"), index=False)

# Table 2: the two folds of the submitted paper (test years 2023 and 2024), every model, every pair, h = 1..12
folds = {"2022-12": "test 2023", "2023-12": "test 2024"}
f = ev2[ev2["origin"].isin(folds) & (ev2["h"] <= 12)].copy()
f["fold"] = f["origin"].map(folds)
f["mase_s"], f["mase_1"] = f["ae"] / f["scale_s"], f["ae"] / f["scale_1"]
tab2 = (f.groupby(P + ["fold", "model"], observed=True)
          .agg(MAE=("ae", "mean"), RMSE=("se", lambda x: float(np.sqrt(x.mean()))),
               MASE_seasonal=("mase_s", "mean"), MASE_one_step=("mase_1", "mean"),
               coverage80=("covered", "mean")).reset_index())
sn = tab2[tab2["model"] == "SeasonalNaive"][P + ["fold", "MAE"]].rename(columns={"MAE": "MAE_SN"})
nv = tab2[tab2["model"] == "Naive"][P + ["fold", "MAE"]].rename(columns={"MAE": "MAE_N"})
tab2 = tab2.merge(sn, on=P + ["fold"]).merge(nv, on=P + ["fold"])
tab2["relMAE_vs_SN"], tab2["relMAE_vs_N"] = tab2["MAE"] / tab2["MAE_SN"], tab2["MAE"] / tab2["MAE_N"]
tab2 = tab2.drop(columns=["MAE_SN", "MAE_N"]).round(4)
tab2.to_csv(os.path.join(OUT_DIR, f"tab2_all_models_two_folds_{MODE}.csv"), index=False)
tab2s = compare(f, "paper folds 2023+2024")
tab2s.to_csv(os.path.join(OUT_DIR, f"tab2_summary_two_folds_{MODE}.csv"), index=False)

# Table 3: per pair, the selected model against both benchmarks, by horizon
s3 = ev2[ev2["model"].isin(["Selected"] + BENCHMARKS)]
keys = s3[s3["model"] == "Selected"][P + ["origin", "h"]].drop_duplicates()
s3 = s3.merge(keys, on=P + ["origin", "h"])
tab3 = s3.pivot_table(index=P + ["bucket"], columns="model", values="ae", aggfunc="mean", observed=True).reset_index()
if "Selected" in tab3:
    tab3["relMAE_vs_SN"] = tab3["Selected"] / tab3["SeasonalNaive"]
    tab3["relMAE_vs_N"]  = tab3["Selected"] / tab3["Naive"]
tab3.round(4).to_csv(os.path.join(OUT_DIR, f"tab3_selected_by_pair_{MODE}.csv"), index=False)

# Table 4: interval coverage and error by forecast step
tab4 = (ev2.groupby(["model", "h"], observed=True).agg(MAE=("ae", "mean"), coverage80=("covered", "mean"),
                                                        n=("ae", "size")).reset_index().round(4))
tab4.to_csv(os.path.join(OUT_DIR, f"tab4_error_and_coverage_by_step_{MODE}.csv"), index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
show = ["horizon", "model", "pairs", "relMAE_vs_SN_gmean", "pct_pairs_beating_SN", "wilcoxon_p_vs_SN",
        "relMAE_vs_N_gmean", "pct_pairs_beating_N", "coverage80_%"]
print("\n=== All origins: each model against the benchmarks ===")
print("relMAE = model MAE / benchmark MAE on the same pairs, origins and months; below 1 means the model is better.")
print("'Selected' exists only from the third origin on, so its rows cover fewer origins than the other rows.")
print(tab1[[c for c in show if c in tab1.columns]].to_string(index=False))
print("\n=== The two folds of the submitted paper (2023 and 2024), horizon 1-12 ===")
print(tab2s[[c for c in show if c in tab2s.columns]].to_string(index=False))

print("\n=== Verdict ===")
for b in BUCKETS:
    r = tab1[(tab1["model"] == "Selected") & (tab1["horizon"] == b)]
    if r.empty or pd.isna(r["relMAE_vs_SN_gmean"].iloc[0]):
        print(f"{b}: no selected-model forecasts at this horizon"); continue
    r = r.iloc[0]
    print(f"{b}: selected model vs Seasonal Naive: relMAE {r['relMAE_vs_SN_gmean']:.3f}, better in "
          f"{r['pct_pairs_beating_SN']:.0f}% of {int(r['pairs_vs_SN'])} pairs (Wilcoxon p = {r['wilcoxon_p_vs_SN']}); "
          f"vs Naive: relMAE {r['relMAE_vs_N_gmean']:.3f}, better in {r['pct_pairs_beating_N']:.0f}% of pairs; "
          f"80% interval coverage {r['coverage80_%']:.0f}%")
print(f"\nFiles written to {OUT_DIR}")
