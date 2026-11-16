# ===== STEP 2b: is the posting share driven by the occupation rather than the skill? =====
# Paste into ONE new cell below the Step 2 cell and run it (Step 2 must have run in this session).
#
# 1) How many skills an ad carries, per occupation and year (all skills, and the 13 programming skills).
# 2) How much of each skill's movement is shared by all 13 skills of the same occupation.
# 3) The same stage rule on a RELATIVE share: postings mentioning skill k divided by the postings-by-skill
#    total of the 13 programming skills in that occupation and month. A change that hits every skill of an
#    occupation equally (tagging density, what gets classified into the occupation) cancels out of this measure.

assert "classify" in globals() and "data" in globals(), "Run the Step 2 cell first."
sk = pd.read_parquet(os.path.join(OUT_DIR, "agg_skills.parquet"))
tt = pd.read_parquet(os.path.join(OUT_DIR, "agg_totals.parquet")).rename(columns={"n_post": "den"})
K2 = ["month", "occupation4d"]
mstr = set(MONTHS.astype(str))
occm = tt.set_index(K2).join([
    sk.groupby(K2)["n_post"].sum().rename("pairs_all"),
    sk[sk["skill"].isin(PROG_SKILLS)].groupby(K2)["n_post"].sum().rename("pairs_basket"),
]).reset_index()
occm = occm[occm["month"].isin(mstr)].copy()
occm["skills_per_posting"] = occm["pairs_all"] / occm["den"]
occm["prog_skills_per_posting"] = occm["pairs_basket"] / occm["den"]
occm["year"] = occm["month"].str[:4]
occm["occ"] = occm["occupation4d"].map(short)
occm.to_csv(os.path.join(OUT_DIR, "audit_skills_per_posting_by_occupation.csv"), index=False)

print("1) Skills per posting, all skills (2018 = Oct-Dec, 2025 = Jan-Mar)")
print(occm.pivot_table(index="occ", columns="year", values="skills_per_posting").round(2).to_string())
print("\n   Programming skills (the 13) per posting")
print(occm.pivot_table(index="occ", columns="year", values="prog_skills_per_posting").round(3).to_string())

last12, prev12 = sorted(mstr)[-12:], sorted(mstr)[-24:-12]
rows = []
for occ in OCCUPATIONS:
    o = occm[occm["occupation4d"] == occ].set_index("month")
    chg = lambda col: (o.loc[last12, col].mean() / o.loc[prev12, col].mean() - 1) * 100
    sk_chg, L12 = [], []
    for skl in PROG_SKILLS:
        if (skl, occ) not in data:
            continue
        s, c = data[(skl, occ)]
        sk_chg.append((s[-12:].mean() / s[-24:-12].mean() - 1) * 100 if s[-24:-12].mean() > 0 else np.nan)
        L12.append(np.log(np.clip(level(s, 12)[11:], 1e-6, None)))
    X = np.array(L12); X = X - X.mean(axis=1, keepdims=True)          # each skill around its own mean
    common = X.mean(axis=0, keepdims=True)                              # what all skills of the occupation share
    r2 = 1 - ((X - common) ** 2).sum() / (X ** 2).sum()
    rows.append({"occupation": short[occ], "skills": len(sk_chg),
                 "all_skills_per_posting_%": round(chg("skills_per_posting"), 1),
                 "prog_skills_per_posting_%": round(chg("prog_skills_per_posting"), 1),
                 "median_skill_share_change_%": round(float(np.nanmedian(sk_chg)), 1),
                 "skills_rising": int(np.sum(np.array(sk_chg) > 0)),
                 "movement_shared_by_all_skills_%": round(r2 * 100, 1)})
print("\n2) Last 12 months against the 12 before, and the share of each skill's movement that is common to the occupation")
print(pd.DataFrame(rows).to_string(index=False))

# ---- 3) relative share within the programming basket ---------------------------------------------
bas = sk[sk["skill"].isin(PROG_SKILLS)].groupby(K2)["n_post"].sum().rename("basket").reset_index()
rel = sk[sk["skill"].isin(PROG_SKILLS)].merge(bas, on=K2)
rel["share"] = rel["n_post"] / rel["basket"] * 100

def rel_series(skill, occ):
    x = rel[(rel["skill"] == skill) & (rel["occupation4d"] == occ)]
    idx = pd.PeriodIndex(x["month"], freq="M")
    return pd.Series(x["share"].to_numpy(float), index=idx).reindex(MONTHS, fill_value=0.0).to_numpy()

rng = np.random.default_rng(2026)
draws_b = pd.DataFrame({
    "W": rng.choice(RANGES["W"], N_DRAWS), "theta": rng.uniform(*RANGES["theta"], N_DRAWS),
    "gamma": rng.uniform(*RANGES["gamma"], N_DRAWS), "delta": rng.uniform(*RANGES["delta"], N_DRAWS),
    "rho": rng.uniform(*RANGES["rho"], N_DRAWS), "k": rng.choice(RANGES["k"], N_DRAWS)})
rows = []
for (skl, occ), (s_post, c) in data.items():
    s = rel_series(skl, occ)
    st = classify(s, c, **BASE, timeline=True)
    end = np.array([classify(s, c, int(p.W), p.theta, p.gamma, p.delta, p.rho, int(p.k)) for p in draws_b.itertuples(index=False)])
    vals, cnt = np.unique(end, return_counts=True)
    L = level(s, BASE["W"]); ip = int(np.nanargmax(L))
    rows.append({"skill": skl, "occupation": occ, "peak_month": str(MONTHS[ip]), "peak_rel_share": round(float(L[ip]), 2),
                 "current_rel_share": round(float(L[-1]), 2), "current_vs_peak": round(float(L[-1] / L[ip]), 3),
                 "change_12m_%": round(float((L[-1] / L[-13] - 1) * 100), 1) if L[-13] > 0 else np.nan,
                 "stage_relative": STAGES[int(st[-1])], "modal_stage": STAGES[int(vals[cnt.argmax()])],
                 "p_modal": round(float(cnt.max() / N_DRAWS), 3),
                 "stage_path": " > ".join(STAGES[x] for i, x in enumerate(st) if x != NYO and (i == 0 or st[i - 1] != x))})
relout = pd.DataFrame(rows)
relout["robust"] = np.where(relout["p_modal"] >= ROBUST, "yes", "no")
post = pd.read_csv(os.path.join(OUT_DIR, "tab5_stages_by_pair.csv"))[["skill", "occupation", "stage"]].rename(columns={"stage": "stage_posting_share"})
relout = relout.merge(post, on=["skill", "occupation"], how="left")
relout.to_csv(os.path.join(OUT_DIR, "tab5b_stages_relative_share.csv"), index=False)

def grid(df, col):
    g = df.assign(occ=df["occupation"].map(short)).pivot(index="skill", columns="occ", values=col)
    return g.reindex(index=PROG_SKILLS, columns=[short[o] for o in OCCUPATIONS]).fillna("not observed")

print("\n3) Stage by pair on the POSTING share (Step 2)")
print(grid(post, "stage_posting_share").to_string())
print("\n   Stage by pair on the RELATIVE share (within the 13 programming skills)")
print(grid(relout, "stage_relative").to_string())
print("\n   Relative share: current level against peak, and 12-month change (%)")
print(grid(relout.assign(x=relout["current_vs_peak"].map("{:.2f}".format) + " / " + relout["change_12m_%"].map("{:+.0f}".format)), "x").to_string())
print("\n   Relative share now (% of the basket), by pair")
print(grid(relout.assign(x=relout["current_rel_share"].map("{:.1f}".format)), "x").to_string())
print("\nRelative share, baseline stage, number of pairs:")
print(relout["stage_relative"].value_counts().to_string())
print(f"Pairs whose stage holds in at least {ROBUST:.0%} of {N_DRAWS} joint draws: {(relout['robust'] == 'yes').sum()} of {len(relout)}")
print(f"Pairs with the same stage under both measures: {(relout['stage_relative'] == relout['stage_posting_share']).sum()} of {len(relout)}")
print("\nStage paths on the relative share:")
print(relout.assign(occupation=relout["occupation"].map(short))[["skill", "occupation", "stage_path", "p_modal"]].to_string(index=False))
print(f"\nFiles written to {OUT_DIR}: tab5b_stages_relative_share.csv, audit_skills_per_posting_by_occupation.csv")
