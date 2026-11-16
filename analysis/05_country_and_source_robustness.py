# ===== STEP 4: do the stages hold by country, with a fixed country mix, for stable sources, and with another basket? =====
# Paste into ONE new cell below the Step 2 cells and run it (Step 2 must have run in this session). Under a minute.
#
# Every variant uses the same stage rule and the same relative share (skill k among the 13 programming skills):
#   All            all countries pooled (the main result)
#   <country>      each of the three largest countries on its own
#   Other          all remaining countries pooled
#   Fixed mix      countries combined with constant weights, so a change in which countries supply ads cannot move a share
#   <stability>    each value of Eurostat's source_stability flag on its own
#   No comp.prog.  the basket without the generic label "computer programming"

assert "classify" in globals() and "data" in globals(), "Run the Step 2 cell first."
rd  = lambda name: pd.read_parquet(os.path.join(OUT_DIR, f"agg_{name}.parquet"))
sk  = rd("skills"); skc = rd("skills_country"); ttc = rd("totals_country"); sks = rd("skills_stab"); tts = rd("totals_stab")
mlist = list(MONTHS.astype(str)); mset = set(mlist)
K2 = ["month", "occupation4d"]
pairs = list(data)
MIN_COUNT_VARIANT = 20          # a pair needs this many postings a month on average inside a variant to be classified

def stages_from(df, skills=PROG_SKILLS, weights=None):
    """df: month, occupation4d, skill, n_post [, country]. Returns {pair: stage} on the relative share."""
    df = df[df["month"].isin(mset) & df["skill"].isin(skills)]
    if weights is None:
        g = df.groupby(K2 + ["skill"], as_index=False)["n_post"].sum()
        g["rel"] = g["n_post"] / g.groupby(K2)["n_post"].transform("sum") * 100
    else:                                                    # fixed country weights within each occupation
        g = df.groupby(K2 + ["country", "skill"], as_index=False)["n_post"].sum()
        g["rel_c"] = g["n_post"] / g.groupby(K2 + ["country"])["n_post"].transform("sum") * 100
        g = g.merge(weights, on=["occupation4d", "country"])
        tot = g.groupby(K2 + ["country"], as_index=False)["w"].first().groupby(K2)["w"].sum().rename("wsum").reset_index()
        g = g.merge(tot, on=K2)
        g["part"] = g["rel_c"] * g["w"] / g["wsum"]
        g = g.groupby(K2 + ["skill"], as_index=False).agg(rel=("part", "sum"), n_post=("n_post", "sum"))
    out = {}
    for (skl, occ) in pairs:
        if skl not in skills:
            continue
        x = g[(g["skill"] == skl) & (g["occupation4d"] == occ)].set_index("month")
        s = x["rel"].reindex(mlist, fill_value=0.0).to_numpy(float)
        c = x["n_post"].reindex(mlist, fill_value=0.0).to_numpy(float)
        if (c > 0).sum() < MIN_MONTHS or c.mean() < MIN_COUNT_VARIANT:
            out[(skl, occ)] = "too few ads"
        else:
            out[(skl, occ)] = STAGES[classify(s, c, **BASE)]
    return out

# ---- country composition over time ---------------------------------------------------------------
tc = ttc[ttc["month"].isin(mset)].copy(); tc["year"] = tc["month"].str[:4]
bycountry = tc.groupby("country")["n_post"].sum().sort_values(ascending=False)
top3 = list(bycountry.index[:3]); top6 = list(bycountry.index[:6])
tc["grp"] = np.where(tc["country"].isin(top6), tc["country"], "other")
comp = tc.pivot_table(index="grp", columns="year", values="n_post", aggfunc="sum")
comp = (comp / comp.sum() * 100).round(1).reindex(top6 + ["other"])
comp.to_csv(os.path.join(OUT_DIR, "audit_country_composition.csv"))
print("1) Share of postings by country and year, % (2018 = Oct-Dec, 2025 = Jan-Mar)")
print(comp.to_string())
print(f"   Three largest countries: {top3}, together {bycountry.iloc[:3].sum() / bycountry.sum():.1%} of postings; "
      f"{bycountry.shape[0]} country codes in total")

ts = tts[tts["month"].isin(mset)].copy(); ts["year"] = ts["month"].str[:4]
stab = ts.pivot_table(index="source_stability", columns="year", values="n_post", aggfunc="sum")
print("\n   Share of postings by source_stability flag and year, %")
print((stab / stab.sum() * 100).round(1).to_string())
stab_vals = list(ts.groupby("source_stability")["n_post"].sum().sort_values(ascending=False).index)

# ---- variants ---------------------------------------------------------------------------------------
V = {"All": stages_from(sk)}
for c in top3:
    V[str(c)] = stages_from(skc[skc["country"] == c])
V["Other"] = stages_from(skc[~skc["country"].isin(top3)])
w = skc[skc["month"].isin(mset)].groupby(["occupation4d", "country"], as_index=False)["n_post"].sum()
w["w"] = w["n_post"] / w.groupby("occupation4d")["n_post"].transform("sum")
V["Fixed mix"] = stages_from(skc, weights=w[["occupation4d", "country", "w"]])
for sv in stab_vals[:4]:
    V[f"src:{sv}"] = stages_from(sks[sks["source_stability"] == sv])
V["No comp.prog."] = stages_from(sk, skills=[x for x in PROG_SKILLS if x != "computer programming"])

T = pd.DataFrame(V); T.index.names = ["skill", "occupation"]
T = T.reset_index()
T.to_csv(os.path.join(OUT_DIR, "tab10_stage_robustness.csv"), index=False)

rows = []
for col in [c for c in T.columns if c not in ("skill", "occupation", "All")]:
    ok = T[col].notna() & (T[col] != "too few ads")
    same = (T.loc[ok, col] == T.loc[ok, "All"])
    opp = (((T.loc[ok, col] == "Growth") & (T.loc[ok, "All"] == "Decline")) | ((T.loc[ok, col] == "Decline") & (T.loc[ok, "All"] == "Growth")))
    rows.append({"variant": col, "pairs_classified": int(ok.sum()), "same_stage_as_All": int(same.sum()),
                 "agreement_%": round(same.mean() * 100, 1) if ok.sum() else np.nan, "opposite_direction": int(opp.sum())})
agree = pd.DataFrame(rows)
agree.to_csv(os.path.join(OUT_DIR, "tab10_stage_robustness_summary.csv"), index=False)

pd.set_option("display.width", 280); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
print("\n2) Agreement of each variant with the main result (same rule, relative share)")
print(agree.to_string(index=False))
cols = [c for c in T.columns if c not in ("skill", "occupation", "All")]
core = [c for c in cols if c != "No comp.prog."]
ncls = (T[core] != "too few ads").sum(axis=1)
nsame = T[core].eq(T["All"], axis=0).sum(axis=1)
T["holds_in"] = nsame.astype(str) + " of " + ncls.astype(str)
print("\n3) Stage by pair in every variant")
v = T.copy(); v["occupation"] = v["occupation"].map(short)
print(v.to_string(index=False))
solid = T[(nsame == ncls) & (ncls >= 3)]
print(f"\nPairs with the same stage in every country, fixed-mix and source variant where they can be classified: "
      f"{len(solid)} of {len(T)}")
print(solid.assign(occupation=solid["occupation"].map(short)).groupby("All")["skill"].count().rename("pairs").to_string())
print(f"\nFiles written to {OUT_DIR}: tab10_stage_robustness.csv, tab10_stage_robustness_summary.csv, audit_country_composition.csv")
