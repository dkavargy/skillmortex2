# ===== STEP 3: ChatGPT date against placebo dates, and an audit of the rapid-obsolescence figure =====
# Paste into ONE new cell below the Step 2 cells and run it (Step 2 must have run in this session). About a minute.
#
# Part A. The paper compares the six months before November 2022 with the six months after. Here the same
#         comparison is repeated at every other month in the window. If November 2022 is special, its changes
#         should stand out from the changes seen at arbitrary dates.
# Part B. The old rapid-obsolescence rule (a drop of 50% within six months of the peak) is applied to all
#         skills, and each flagged skill is checked: did its level fall, or did a short spike simply end?

assert "classify" in globals() and "data" in globals(), "Run the Step 2 cell first."
# reload the Step 0 tables: the Step 1 loop reuses the name `sk` for its loop variable
sk = pd.read_parquet(os.path.join(OUT_DIR, "agg_skills.parquet"))
tt = pd.read_parquet(os.path.join(OUT_DIR, "agg_totals.parquet")).rename(columns={"n_post": "den"})
EVENT = "2022-11"
K2 = ["month", "occupation4d"]
mlist = list(MONTHS.astype(str)); n = len(mlist); e = mlist.index(EVENT)

# ---------------- Part A ---------------------------------------------------------------------------
prog = sk[sk["skill"].isin(PROG_SKILLS)].copy()
prog["rel"] = prog["n_post"] / prog.groupby(K2)["n_post"].transform("sum") * 100
prog = prog.merge(tt, on=K2); prog["post"] = prog["n_post"] / prog["den"] * 100

def mat(col):
    out = {}
    for (skl, occ) in data:
        x = prog[(prog["skill"] == skl) & (prog["occupation4d"] == occ)].set_index("month")[col]
        out[(skl, occ)] = x.reindex(mlist, fill_value=0.0).to_numpy(float)
    return pd.DataFrame(out, index=mlist)

def deltas(S):
    """Percentage change, six months after against six months before, at every possible date."""
    rows = {}
    for t in range(6, n - 6):
        pre, post = S.iloc[t - 6:t].mean(), S.iloc[t + 1:t + 7].mean()
        rows[mlist[t]] = (post / pre.replace(0, np.nan) - 1) * 100
    return pd.DataFrame(rows).T

S_rel, S_post = mat("rel"), mat("post")
D, Dp = deltas(S_rel), deltas(S_post)
placebo = [m for m in D.index if abs(mlist.index(m) - e) > 6]          # windows that do not touch the event
act, plc = D.loc[EVENT], D.loc[placebo]
pretrend = (S_rel.iloc[e - 6:e].mean() / S_rel.iloc[e - 12:e - 6].mean() - 1) * 100
pval = (plc.abs() >= act.abs()).mean()

pairsA = pd.DataFrame({
    "change_at_event_%": act.round(1), "change_before_event_%": pretrend.round(1),
    "event_minus_before_pp": (act - pretrend).round(1),
    "placebo_p05_%": plc.quantile(0.05).round(1), "placebo_p95_%": plc.quantile(0.95).round(1),
    "placebo_typical_abs_%": plc.abs().median().round(1),
    "share_of_placebo_dates_as_large": pval.round(3),
    "change_at_event_posting_share_%": Dp.loc[EVENT].round(1),
})
pairsA.index.names = ["skill", "occupation"]
pairsA = pairsA.reset_index()
pairsA["old_rule_flag(>=+5%)"] = np.where(pairsA["change_at_event_%"] >= 5, "yes", "")
pairsA["unusual(p<0.10)"] = np.where(pairsA["share_of_placebo_dates_as_large"] < 0.10, "yes", "")
pairsA.to_csv(os.path.join(OUT_DIR, "tab8_chatgpt_placebo_by_pair.csv"), index=False)

bydate = pd.DataFrame({
    "mean_abs_change_%": D.abs().mean(axis=1), "median_abs_change_%": D.abs().median(axis=1),
    "pairs_up_5%": (D >= 5).sum(axis=1), "pairs_down_5%": (D <= -5).sum(axis=1),
    "pairs_moving_5%": (D.abs() >= 5).sum(axis=1), "pairs_moving_10%": (D.abs() >= 10).sum(axis=1)}).round(2)
bydate["kind"] = ["EVENT" if m == EVENT else ("placebo" if m in placebo else "overlaps event") for m in bydate.index]
bydate.to_csv(os.path.join(OUT_DIR, "tab8_chatgpt_placebo_by_date.csv"))
pb = bydate[bydate["kind"] == "placebo"]

def rank_line(col, label):
    v = bydate.loc[EVENT, col]
    higher = int((pb[col] >= v).sum())
    print(f"  {label:38s} event date {v:6.1f}   placebo dates: median {pb[col].median():6.1f}, "
          f"5th-95th pct {pb[col].quantile(.05):.1f}-{pb[col].quantile(.95):.1f}   "
          f"placebo dates at least as high: {higher} of {len(pb)}")

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
print(f"PART A. Relative share, {len(data)} pairs. Event {EVENT}: six months before against six months after.")
print(f"Placebo dates: {len(pb)} months from {placebo[0]} to {placebo[-1]}, leaving out the 13 months around the event.\n")
print("How November 2022 compares with an arbitrary date:")
rank_line("mean_abs_change_%", "mean absolute change (%)")
rank_line("median_abs_change_%", "median absolute change (%)")
rank_line("pairs_up_5%", "pairs rising 5% or more (old rule)")
rank_line("pairs_moving_5%", "pairs moving 5% or more, either way")
rank_line("pairs_moving_10%", "pairs moving 10% or more")
nunus = int((pairsA["unusual(p<0.10)"] == "yes").sum())
print(f"\nPairs whose change at the event is larger than at 90% of placebo dates: {nunus} of {len(pairsA)} "
      f"(about {0.10 * len(pairsA):.0f} expected by chance)")
print("\nSame comparison at each November (same months of the year before and after):")
print(bydate.loc[[m for m in bydate.index if m.endswith("-11")]].to_string())
print("\nThe ten dates with the largest mean absolute change:")
print(bydate.sort_values("mean_abs_change_%", ascending=False).head(10).to_string())
v = pairsA.copy(); v["occupation"] = v["occupation"].map(short)
print("\nBy pair (relative share unless stated):")
print(v.drop(columns=["placebo_typical_abs_%"]).to_string(index=False))

# ---------------- Part B ---------------------------------------------------------------------------
FIG9 = ["consultation methods", "identify customer's needs", "data protection", "ict communications protocols",
        "perform business analysis", "abap", "quality control systems", "microsoft access", "ict system integration",
        "recruit employees", "pricing strategies", "scala"]
pool = sk[sk["month"].isin(set(mlist))].groupby(["month", "skill"], as_index=False)["n_post"].sum()
pool["share"] = pool["n_post"] / pool.groupby("month")["n_post"].transform("sum") * 100
vol = pool.groupby("skill")["n_post"].sum()
cand = vol[vol >= 50000].index
W = pool[pool["skill"].isin(set(cand) | set(FIG9))].pivot(index="month", columns="skill", values="share").reindex(mlist).fillna(0.0)

rowsB = []
for skl in W.columns:
    s = W[skl].to_numpy(); ip = int(s.argmax()); pk = s[ip]
    if pk <= 0:
        continue
    lo6 = s[ip:ip + 7].min()
    drop = (pk - lo6) / pk
    later = s[ip + 7:].max() / pk if ip + 7 < n else np.nan
    flagged = bool(drop >= 0.50 and (s > 0).sum() >= 10 and (np.isnan(later) or later <= 0.70) and ip + 6 < n)
    before = np.median(s[max(ip - 18, 0):ip - 6]) if ip - 6 >= 6 else np.nan
    after  = np.median(s[ip + 7:ip + 19]) if ip + 12 < n else np.nan
    ratio  = after / before if (before and before > 0 and not np.isnan(after)) else np.nan
    spike_months = int((s[max(ip - 6, 0):ip + 7] > 1.5 * before).sum()) if before and before > 0 else np.nan
    if ip < 6:
        kind = "peak in first 6 months of data"
    elif np.isnan(ratio):
        kind = "too close to an end of the data"
    elif ratio >= 0.8:
        kind = "spike: level afterwards is back to before"
    else:
        kind = "level fell"
    rowsB.append({"skill": skl, "in_fig9": "yes" if skl in FIG9 else "", "flagged_by_old_rule": "yes" if flagged else "",
                  "peak_month": mlist[ip], "months_from_event": ip - e, "drop_in_6m_%": round(drop * 100, 0),
                  "peak_vs_level_before": round(pk / before, 1) if before and before > 0 else np.nan,
                  "months_above_1.5x_before": spike_months,
                  "level_after_vs_before": round(ratio, 2) if not np.isnan(ratio) else np.nan, "reading": kind})
B = pd.DataFrame(rowsB)
B.to_csv(os.path.join(OUT_DIR, "tab9_rapid_obsolescence_audit.csv"), index=False)
fl = B[B["flagged_by_old_rule"] == "yes"]
print(f"\n\nPART B. Skills with at least 50,000 postings-by-skill, all four occupations pooled: {len(cand)}. "
      f"Share = skill's part of all skill mentions in the month.")
print(f"Flagged by the old rule (drop of 50% within six months of the peak, no later return above 70% of peak): {len(fl)}")
if len(fl):
    print(fl["reading"].value_counts().to_string())
    print(f"Peak before {EVENT}: {(fl['months_from_event'] < 0).sum()};  at or after: {(fl['months_from_event'] >= 0).sum()}")
    print("\nFlagged skills:")
    print(fl.sort_values("drop_in_6m_%", ascending=False).drop(columns=["flagged_by_old_rule"]).to_string(index=False))
print("\nThe twelve skills shown in Fig. 9 of the submitted paper:")
print(B[B["in_fig9"] == "yes"].drop(columns=["in_fig9"]).to_string(index=False))
missing = [x for x in FIG9 if x not in set(B["skill"])]
if missing:
    print("Not found in the data under that name:", missing)
print(f"\nFiles written to {OUT_DIR}: tab8_chatgpt_placebo_by_pair.csv, tab8_chatgpt_placebo_by_date.csv, tab9_rapid_obsolescence_audit.csv")
