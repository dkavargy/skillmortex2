import sys, calendar, numpy as np, pandas as pd
sys.path.insert(0, "../figures")
import smxfig as S
sk, tt, prog = S.base_tables()
mset = set(S.MLIST); L = S.SKILL_LABEL; O = S.OCC_SHORT
out = []
def tab(caption, label, spec, head, rows, note=None, size="\\small", env="table"):
    out.append(f"\\begin{{{env}}}[!htbp]\n\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\\\\\n\\midrule")
    out.extend(r if r.startswith("\\") else r + "\\\\" for r in rows)
    out.append("\\bottomrule\n\\end{tabular}")
    if note: out.append(f"\\par\\vspace{{3pt}}\\begin{{minipage}}{{0.97\\textwidth}}\\footnotesize\\emph{{Note.}} {note}\\end{{minipage}}")
    out.append(f"\\end{{{env}}}\n")
pct = lambda v, d=1: f"{v:.{d}f}"

# ---------- T1 reused vs new
rows = ["Data & 2.3 million Stack Overflow questions, 2008--2024 (supply side) & 12.9 million online job advertisements, October 2018--March 2025 (demand side)",
        "Unit of analysis & Skill tag & Skill within an occupation (48 pairs)",
        "Life-cycle metaphor and stage names & Introduced & Reused as a heuristic",
        "Stage detection & Date of birth and peak activity date, descriptive & New: one explicit rule; joint variation of all thresholds",
        "Rapid-obsolescence rule & Introduced & Reused, tested, not retained (Appendix)",
        "Before/after comparison around ChatGPT & Introduced & Reused, tested against placebo dates, not supported",
        "Forecast evaluation & None & New: seven models against two naive benchmarks",
        "Country and source checks & None & New"]
tab("Elements reused from Kavargyris et al.\\ (2025) and elements that are new in this paper.", "tab:reused",
    "@{}p{0.24\\textwidth}p{0.33\\textwidth}p{0.36\\textwidth}@{}", " & Kavargyris et al.\\ (2025) & This paper", rows)

# ---------- T2 data by occupation, with the cap
a = S.tab("audit_files.csv"); fn = a["file"].str.split("__").str[-1].str.replace(".csv", ""); low = fn.str.lower()
a["occ"] = np.select([low.str.contains("software"), low.str.contains("systems"), low.str.contains("web"), low.str.contains("applications")], S.OCCUPATIONS, "?")
a["ym"] = fn.str.extract(r"(\d{4}-\d{2})")[0]
mon = a.dropna(subset=["ym"]).groupby(["occ", "ym"], as_index=False)["rows"].sum()
mon["rows_day"] = mon["rows"] / [calendar.monthrange(*map(int, y.split("-")))[1] for y in mon["ym"]]
half = a[a["ym"].isna()]; hy = fn[half.index].str.extract(r"(\d{4})_H([12])"); extra = []
for (i, r), (y, h) in zip(half.iterrows(), hy.itertuples(index=False)):
    ms = [f"{y}-{m:02d}" for m in (range(1, 7) if h == "1" else range(7, 13))]
    nd = sum(calendar.monthrange(int(y), int(m[-2:]))[1] for m in ms)
    extra += [{"occ": r["occ"], "ym": m, "rows_day": r["rows"] / nd} for m in ms]
mon = pd.concat([mon, pd.DataFrame(extra)], ignore_index=True); mon = mon[mon["ym"].isin(mset)]
atcap = lambda g: float((((g >= 9700) & (g <= 10001)) | ((g >= 14500) & (g <= 15001))).mean())
w = sk[sk["month"].isin(mset)]; t = tt[tt["month"].isin(mset)]
ads = t.groupby("occupation4d")["den"].sum(); men = w.groupby("occupation4d")["n_mentions"].sum()
mm = w.groupby(["occupation4d", "month"])["n_mentions"].sum() / t.set_index(["occupation4d", "month"])["den"]
rows = []
for o in S.OCCUPATIONS:
    q = mm.loc[o].quantile([.1, .9]); cap = atcap(mon.loc[mon["occ"] == o, "rows_day"])
    rows.append(f"{o} & {ads[o]:,.0f} & {ads[o] / ads.sum() * 100:.1f} & {men[o] / 1e6:.1f} & {q.iloc[0]:.0f}--{q.iloc[1]:.0f} & {cap * 100:.0f}")
rows += ["\\midrule", f"Total & {ads.sum():,.0f} & 100.0 & {men.sum() / 1e6:.1f} & & "]
tab("Advertisements retrieved by occupation, October 2018--March 2025.", "tab:data", "@{}lrrrrr@{}",
    "Occupation & Advertisements & Share (\\%) & Rows (million) & Rows per ad & At the cap (\\%)", rows,
    "Occupations are ISCO-08 unit groups. A row is one skill mentioned in one advertisement. At the cap: share of months at the export limit. Rows per ad: 10th to 90th percentile of the monthly values. The export returned at most 10,000 rows per occupation and day (15,000 for Systems analysts until December 2022); a month is at the cap when its mean number of rows per day is within 3\\% of that limit. An advertisement is counted when at least one of its rows was retrieved.")
T2 = (ads.to_dict(), men.sum())

# ---------- T3 stage rule, T4 thresholds
rows = ["Birth & First 12 months after the first month in which the skill is observed; not assigned when the skill is present in the first month of the data (left-censored)",
        "Growth & $r_t \\ge \\theta$ and $g_t \\ge \\gamma$",
        "Maturity & $r_t \\ge \\theta$ and $g_t < \\gamma$",
        "Decline & $\\delta \\le r_t < \\theta$ and the Revival condition does not hold",
        "Revival & $\\delta \\le r_t < \\theta$, $g_t \\ge \\gamma$ and $L_t \\ge (1+\\rho)\\,T_t$",
        "Death & $r_t < \\delta$, or fewer than five advertisements a month on average over the last $W$ months"]
tab("The stage rule. One rule is applied to every skill--occupation pair and uses observed data only.", "tab:rule", "@{}lp{0.80\\textwidth}@{}", "Stage & Condition in month $t$", rows,
    "$s_t$: relative share. $L_t$: level, the median of $s$ over the last $W$ months. $P_t$: running peak of $L$. $r_t = L_t/P_t$. $g_t = L_t/L_{t-12}-1$. $T_t$: lowest level since the peak. Conditions are checked in the order Death, Growth, Maturity, Revival, Decline. A new stage is recorded only after it has held for $k$ consecutive months.")
# agreement by threshold, recomputed on the relative share (same seed and draw order as the notebook)
rng = np.random.default_rng(2026); N = S.N_DRAWS; RG = S.RANGES
draws = pd.DataFrame({"W": rng.choice(RG["W"], N), "theta": rng.uniform(*RG["theta"], N), "gamma": rng.uniform(*RG["gamma"], N),
                      "delta": rng.uniform(*RG["delta"], N), "rho": rng.uniform(*RG["rho"], N), "k": rng.choice(RG["k"], N)})
AP = {k: d for k, d in S.all_pairs().items() if d["observed"]}
res = np.array([[S.classify(d["s"], d["c"], int(p.W), p.theta, p.gamma, p.delta, p.rho, int(p.k)) for d in AP.values()] for p in draws.itertuples(index=False)])
basel = np.array([S.classify(d["s"], d["c"], **S.BASE) for d in AP.values()])
agree = pd.Series((res == basel).mean(axis=1)); AGREE_ALL = float(agree.mean()); s7 = {}
for p in draws.columns:
    grp = draws[p] if draws[p].nunique() <= 5 else pd.qcut(draws[p], 5)
    q = agree.groupby(grp.to_numpy(), observed=True).mean()
    s7[p] = {"baseline": S.BASE[p], "agreement_lowest_%": q.min() * 100, "agreement_highest_%": q.max() * 100}
s7 = pd.DataFrame(s7).T; print("agreement overall", round(AGREE_ALL, 3)); print(s7.round(1))
P7 = [("W", "$W$", "Window of the level (months)", "9, 12, 15"), ("theta", "$\\theta$", "Level relative to peak that separates Growth/Maturity from Decline", "0.70--0.90"),
      ("gamma", "$\\gamma$", "12-month change that separates Growth from Maturity", "0.05--0.15"), ("delta", "$\\delta$", "Level relative to peak below which a skill is in Death", "0.05--0.20"),
      ("rho", "$\\rho$", "Rise above the lowest level since the peak required for Revival", "0.10--0.30"), ("k", "$k$", "Consecutive months before a stage change is recorded", "2, 3, 4")]
rows = [f"{sym} & {desc} & {s7.loc[p, 'baseline']:g} & {rng} & {s7.loc[p, 'agreement_lowest_%']:.0f}--{s7.loc[p, 'agreement_highest_%']:.0f}" for p, sym, desc, rng in P7]
tab("Thresholds of the stage rule: baseline value, range used in the 2,000 joint draws, and agreement with the baseline stage.", "tab:thresholds",
    "@{}lp{0.50\\textwidth}rrr@{}", " & Meaning & Baseline & Range & Agreement (\\%)", rows,
    f"Agreement: share of pair-draws with the same stage as under the baseline, lowest and highest value across the fifths of the range of that threshold (for $W$ and $k$: across its three values). All six thresholds vary together in every draw; over all draws and pairs the agreement is {AGREE_ALL * 100:.0f}\\%.")

# ---------- T5 results by pair
r5 = S.tab("tab5b_stages_relative_share.csv"); g = pd.read_csv(__import__("os").path.join(S.OUT, "evidence_grades_by_pair.csv"))
r5 = r5.merge(g[["skill", "occupation", "same", "n", "grade"]], on=["skill", "occupation"])
prob = S.stage_probabilities() if hasattr(S, "stage_probabilities") else None
order = S.skill_order()[::-1]; rows = []
GR = {"consistent": "firm", "pooled": "pooled only", "uncertain": "uncertain"}
for i, skl in enumerate(order):
    if i: rows.append("\\addlinespace[2pt]")
    for o in S.OCCUPATIONS:
        x = r5[(r5["skill"] == skl) & (r5["occupation"] == o)]
        if x.empty:
            rows.append(f"{L[skl] if o == S.OCCUPATIONS[0] else ''} & {O[o]} & \\multicolumn{{8}}{{c}}{{not observed}}"); continue
        x = x.iloc[0]
        pstage = float(prob.loc[(skl, o), x["stage_relative"]]) if prob is not None else x["p_modal"]
        rows.append(f"{L[skl] if o == S.OCCUPATIONS[0] else ''} & {O[o]} & {x['current_rel_share']:.1f} & {pd.Period(x['peak_month'], 'M').strftime('%b %Y')} & {x['current_vs_peak']:.2f} & {(0 if abs(x['change_12m_%']) < 0.5 else x['change_12m_%']):+.0f} & {x['stage_relative']} & {pstage * 100:.0f} & {x['same']}/{x['n']} & {GR[x['grade']]}".replace("+-", "$-$").replace("& -", "& $-$"))
tab("Stage of each skill--occupation pair in March 2025 (relative share).", "tab:pairs", "@{}llrlrrlrrl@{}",
    "Skill & Occupation & Level & Peak & $L/P$ & $g$ & Stage & Draws & Re-runs & Evidence", rows,
    "Level: 12-month median of the relative share in March 2025 (\\%). $L/P$: level divided by its peak. $g$: change of the level over 12 months (\\%). Draws: share of the 2,000 joint threshold draws that give the baseline stage (\\%). Re-runs: number of the seven country, fixed-weight and source re-runs with the same stage, out of those in which the pair can be classified (a re-run in Death is counted as agreeing with Decline). Evidence: firm = at least 85\\% of draws and at most one re-run differs; pooled only = at least 85\\% of draws, two or more re-runs differ; uncertain = under 85\\% of draws.",
    size="\\scriptsize\\setlength{\\tabcolsep}{4.5pt}")

# ---------- T6 summary by occupation and by evidence
rows = []
for st in ["Growth", "Maturity", "Decline"]:
    c = [int(((r5["stage_relative"] == st) & (r5["occupation"] == o)).sum()) for o in S.OCCUPATIONS]
    e = [int(((r5["stage_relative"] == st) & (r5["grade"] == k)).sum()) for k in ("consistent", "pooled", "uncertain")]
    rows.append(f"{st} & " + " & ".join(map(str, c)) + f" & {sum(c)} & " + " & ".join(map(str, e)))
c = [int((r5["occupation"] == o).sum()) for o in S.OCCUPATIONS]; e = [int((r5["grade"] == k).sum()) for k in ("consistent", "pooled", "uncertain")]
rows += ["\\midrule", "Observed pairs & " + " & ".join(map(str, c)) + f" & {sum(c)} & " + " & ".join(map(str, e)), "Not observed & " + " & ".join(str(13 - x) for x in c) + f" & {52 - sum(c)} & & & "]
tab("Number of pairs by stage in March 2025, by occupation and by strength of evidence.", "tab:summary", "@{}lrrrrr@{\\hspace{14pt}}rrr@{}",
    "Stage & Software & Systems & Web & Applic. & All & Firm & Pooled only & Uncertain", rows, "No pair is in Birth, Revival or Death in March 2025. Evidence classes as in Table~\\ref{tab:pairs}.")
T6 = r5["stage_relative"].value_counts().to_dict()

# ---------- T7 forecast evaluation
f1 = S.tab("tab1_models_by_horizon_relative.csv")
MOD = [("Naive", "Naive (last value)"), ("SeasonalNaive", "Seasonal naive"), ("HW_Additive_Damped", "Holt--Winters, additive damped"), ("HW_Multiplicative", "Holt--Winters, multiplicative"),
       ("SARIMA(0,1,1)(0,1,1)12", "SARIMA(0,1,1)(0,1,1)$_{12}$"), ("SARIMA(1,1,1)(1,1,1)12", "SARIMA(1,1,1)(1,1,1)$_{12}$"), ("Prophet", "Prophet"), ("Selected", "Selected on the previous year")]
rows = []
for m, lab in MOD:
    cells = []
    for h in ("h01-12", "h13-24", "h25-36"):
        x = f1[(f1["model"] == m) & (f1["horizon"] == h)].iloc[0]
        f = lambda v: "1.00" if (pd.isna(v)) else f"{v:.2f}"
        cells += [f(x["relMAE_vs_N_gmean"]), f(x["relMAE_vs_SN_gmean"]), f"{x['coverage80_%']:.0f}"]
    if m == "Selected": rows.append("\\midrule")
    rows.append(lab + " & " + " & ".join(cells))
tab("Out-of-sample forecast error relative to the two benchmarks, and coverage of the 80\\% prediction interval. Relative share, 48 pairs, seven forecast origins (December 2020 to December 2023).",
    "tab:forecast", "@{}l@{\\hspace{10pt}}rrr@{\\hspace{14pt}}rrr@{\\hspace{14pt}}rrr@{}",
    " & \\multicolumn{3}{c}{1--12 months} & \\multicolumn{3}{c}{13--24 months} & \\multicolumn{3}{c}{25--36 months}\\\\\n\\cmidrule(r){2-4}\\cmidrule(r){5-7}\\cmidrule{8-10}\nModel & vs N & vs SN & Cov. & vs N & vs SN & Cov. & vs N & vs SN & Cov.", rows,
    "vs N, vs SN: mean absolute error of the model divided by that of the naive (N) and the seasonal naive (SN) forecast, geometric mean over pairs; values above 1 mean the model is worse than the benchmark. Cov.: share of outcomes inside the 80\\% prediction interval (\\%). Selected: for each pair and origin, the model with the lowest error over the preceding 12 months. Only the first three origins cover the full 36 months.")

# ---------- T8 placebo
bd = S.tab("tab8_chatgpt_placebo_by_date.csv"); bp = S.tab("tab8_chatgpt_placebo_by_pair.csv")
ev = bd[bd["kind"] == "EVENT"].iloc[0]; pl = bd[bd["kind"] == "placebo"]; rows = []
for col, lab in [("mean_abs_change_%", "Mean absolute change over the 48 pairs (\\%)"), ("pairs_up_5%", "Pairs rising by 5\\% or more")]:
    q = pl[col].quantile([.05, .5, .95]); n_ge = int((pl[col] >= ev[col]).sum())
    fmt = (lambda v: f"{v:.1f}") if "abs" in col else (lambda v: f"{v:.0f}")
    rows.append(f"{lab} & {fmt(ev[col])} & {fmt(q.iloc[1])} & {fmt(q.iloc[0])}--{fmt(q.iloc[2])} & {n_ge} of {len(pl)}")
un = int((bp["unusual(p<0.10)"] == "yes").sum())
rows.append(f"Pairs with a change larger than at 90\\% of their own placebo dates & {un} of {len(bp)} & \\multicolumn{{3}}{{l}}{{about {len(bp) * 0.1:.0f} expected by chance}}")
tab("Change in relative share around November 2022 compared with placebo dates.", "tab:placebo", "@{}p{0.50\\textwidth}rrrr@{}",
    " & & \\multicolumn{3}{c}{Placebo dates}\\\\\n\\cmidrule{3-5}\n & Nov.\\ 2022 & Median & 5--95\\% & As high", rows,
    f"As high: number of placebo dates with a value at least as high as in November 2022. Change: mean relative share in the six months after a date divided by the mean in the six months before. Placebo dates: the {len(pl)} months between April 2019 and September 2024 that are more than six months away from November 2022.")
T8 = (ev.to_dict(), un, len(pl))

# ---------- T9 spike audit
au = S.tab("tab9_rapid_obsolescence_audit.csv"); fl = au[au["flagged_by_old_rule"] == "yes"].copy()
RD = {"spike: level afterwards is back to before": "Spike; level back", "level fell": "Level fell", "too close to an end of the data": "Too recent to tell"}
fl["o"] = fl["reading"].map({k: i for i, k in enumerate(RD)}); fl = fl.sort_values(["o", "peak_month"]); rows = []
NM = {"abap": "ABAP", "ict communications protocols": "ICT communications protocols", "microsoft access": "Microsoft Access"}
for x in fl.itertuples():
    nm = NM.get(x.skill, x.skill[0].upper() + x.skill[1:])
    f = lambda v, d=2: "--" if pd.isna(v) else f"{v:.{d}f}"
    rows.append(f"{nm} & {pd.Period(x.peak_month, 'M').strftime('%b %Y')} & {x._6:.0f} & {f(x.peak_vs_level_before, 1)} & {f(x.level_after_vs_before)} & {RD[x.reading]}")
tab(f"The {len(fl)} skills flagged by the rapid-obsolescence rule of the first submission, out of {len(au)} skills checked.", "tab:spikes", "@{}llrrrl@{}",
    "Skill & Peak & Fall (\\%) & Peak/before & After/before & Reading", rows,
    "Share of the skill in all skill mentions, four occupations pooled. Fall: largest fall within six months of the peak. Peak/before and After/before: the peak and the level after it, divided by the level before it. Level before: median of months 18 to 7 before the peak. Level after: median of months 7 to 18 after the peak.")

# ---------- T10 country composition, T11 agreement by variant
cc = S.tab("audit_country_composition.csv"); CN = {"United Kingdom": "United Kingdom", "DEUTSCHLAND": "Germany", "France": "France", "Nederland": "Netherlands", "ITALIA": "Italy", "Polska": "Poland", "other": "Other countries"}
yrs = [c for c in cc.columns if c != "grp"]
rows = [CN[x["grp"]] + " & " + " & ".join(f"{x[y]:.1f}" for y in yrs) for _, x in cc.iterrows()]
tab("Share of advertisements by country and year (\\%).", "tab:countries", "@{}l" + "r" * len(yrs) + "@{}", "Country & " + " & ".join(yrs), rows, "2018: October to December. 2025: January to March.")
ag = S.tab("tab10_stage_robustness_summary.csv").set_index("variant")
VR = [("United Kingdom", "United Kingdom only"), ("DEUTSCHLAND", "Germany only"), ("France", "France only"), ("Other", "All other countries"), ("Fixed mix", "All countries, fixed country weights"),
      ("src:stable", "Stable sources only"), ("src:unstable", "Unstable sources only"), ("No comp.prog.", "Basket without ``computer programming''")]
rows = [f"{lab} & {int(ag.loc[k, 'pairs_classified'])} & {int(ag.loc[k, 'same_stage_as_All'])} & {ag.loc[k, 'agreement_%']:.0f} & {int(ag.loc[k, 'opposite_direction'])}" for k, lab in VR]
tab("The stage rule re-run on parts of the data: agreement with the stages obtained from all advertisements.", "tab:reruns", "@{}lrrrr@{}",
    "Re-run & Pairs classified & Same stage & Agreement (\\%) & Opposite direction", rows,
    "A pair is classified in a re-run when it has at least 36 months with data and at least 20 advertisements a month on average. Opposite direction: Growth in one and Decline in the other.")

open("tables_body.tex", "w").write("\n".join(out))
print(T2, T6, T8)
