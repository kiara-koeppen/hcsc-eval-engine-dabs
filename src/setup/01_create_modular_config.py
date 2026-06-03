# Databricks notebook source
# MAGIC %md
# MAGIC # Setup (Modular Track): Registry config + multi-period data
# MAGIC
# MAGIC This sets up the **modular / multi-team** variant of the evaluation engine. It is
# MAGIC ADDITIVE: it does not touch the original demo's `study_config` / `cohorts`. It creates:
# MAGIC
# MAGIC | Table | Purpose |
# MAGIC |-------|---------|
# MAGIC | `study_config_modular` | One row per study. Instead of method *strings*, each row names the **leaf notebook** to run for each stage (`feature_nb`, `matching_nb`, `model_nb`). This is the "registry": teams own their own notebooks and register them here. |
# MAGIC | `cohorts` (append) | Single-period treated/control patients for the standard-family studies (same shape as the original demo). |
# MAGIC | `cohorts_multiperiod` | Long-format patient x period outcomes for the LME study (one baseline + several performance periods). This is the structurally-different case the team raised. |
# MAGIC
# MAGIC ### Why a registry instead of one big parameterized notebook
# MAGIC The team keeps notebooks **separate on purpose**: multiple model developers work in
# MAGIC parallel, and merging everything into a single notebook causes breakage. The registry
# MAGIC pattern keeps every method/model in its **own** notebook (its own file, its own owner)
# MAGIC while the orchestration DAG stays flat. A stage dispatcher reads the notebook name from
# MAGIC config and runs it with `dbutils.notebook.run(...)`. Adding a model = drop in a new
# MAGIC notebook + add one config row. No shared file to merge, no DAG edit.

# COMMAND ----------

# MAGIC %md ### Parameters (always parameterize catalog/schema — never hardcode)

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
print(f"Target: {catalog}.{schema}")

# Schema only (see note in 00_create_config_and_data: CREATE CATALOG can error on
# Default-Storage metastores even when the catalog exists).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md ### 1. Registry config table
# MAGIC Each row names the leaf notebooks for its stages. `model_family` is informational
# MAGIC (it documents which flow a study belongs to); the actual dispatch is by notebook name.
# MAGIC
# MAGIC - `M_ATT_001`, `M_DID_002` — standard family: shared `feature_standard` + `matching_standard`, different **model** notebooks (one per team: `model_att`, `model_did`).
# MAGIC - `M_STRAT_003` — shows a **third team plugging in** a new matching method (`matching_stratified`) with zero changes to anyone else's code.
# MAGIC - `M_LME_004` — the **structurally different** study: its own `feature_lme` (multi performance periods) and `model_lme`, but it still reuses `matching_standard`. Different where it must be, shared where it can be.

# COMMAND ----------

from pyspark.sql import Row

# columns: study_id, study_name, vendor, model_family,
#          feature_nb, matching_nb, matching_method, model_nb, min_quality_score
registry_rows = [
    ("M_ATT_001",   "Diabetes Care Mgmt (ATT)",    "VendorA Health",   "standard",
     "feature_standard", "matching_standard", "exact", "model_att", 0.80),
    ("M_DID_002",   "CHF Remote Monitoring (DID)",  "VendorB Cardio",   "standard",
     "feature_standard", "matching_standard", "knn",   "model_did", 0.80),
    ("M_STRAT_003", "Maternity Support (ATT/strat)","VendorC Maternal", "standard",
     "feature_standard", "matching_stratified", "stratified", "model_att", 0.75),
    ("M_LME_004",   "Wellness Longitudinal (LME)",  "VendorD Wellness", "lme",
     "feature_lme",      "matching_standard", "knn",   "model_lme", 0.80),
]

registry_df = spark.createDataFrame([
    Row(study_id=r[0], study_name=r[1], vendor=r[2], model_family=r[3],
        feature_nb=r[4], matching_nb=r[5], matching_method=r[6], model_nb=r[7],
        min_quality_score=float(r[8]))
    for r in registry_rows
])
registry_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("study_config_modular")
display(spark.table("study_config_modular"))

# COMMAND ----------

# MAGIC %md ### 2. Single-period cohorts for the standard-family studies
# MAGIC Same generator/shape as the original demo, written for the modular study_ids so the
# MAGIC modular track is self-contained. Appended into the shared `cohorts` table.

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# (study_id, n_per_arm, true_effect, missing_rate)
single_specs = {
    "M_ATT_001":   (1200, -450.0, 0.02),
    "M_DID_002":   (900,  -1.20,  0.03),
    "M_STRAT_003": (1500, -300.0, 0.04),
}

def make_arm(study_id, n, treated, true_effect, missing_rate):
    # Treated patients are sicker on average (confounding) so naive comparison is
    # biased and matching/weighting actually matters.
    shift = 0.6 if treated else 0.0
    age = rng.normal(58 + 4 * shift, 12, n)
    risk_score = rng.normal(1.5 + 0.5 * shift, 0.6, n).clip(0.1, None)
    prior_cost = rng.normal(8000 + 2500 * shift, 3000, n).clip(0, None)
    pre_outcome = 5000 + 0.4 * prior_cost + 600 * risk_score + rng.normal(0, 1500, n)
    post_outcome = pre_outcome - 200 + (true_effect if treated else 0.0) + rng.normal(0, 1500, n)
    df = pd.DataFrame({
        "study_id": study_id, "treatment": int(treated),
        "age": age, "risk_score": risk_score, "prior_cost": prior_cost,
        "pre_outcome": pre_outcome, "post_outcome": post_outcome,
    })
    if missing_rate > 0:
        mask = rng.random(n) < missing_rate
        df.loc[mask, "risk_score"] = np.nan
    return df

frames = []
for sid, (n, eff, miss) in single_specs.items():
    frames.append(make_arm(sid, n, treated=True,  true_effect=eff, missing_rate=miss))
    frames.append(make_arm(sid, n, treated=False, true_effect=eff, missing_rate=miss))

single = pd.concat(frames, ignore_index=True)

# Keep patient_id globally unique vs whatever is already in `cohorts`.
existing_max = (spark.table(f"{catalog}.{schema}.cohorts").selectExpr("max(patient_id) m").collect()[0]["m"]
                if spark.catalog.tableExists(f"{catalog}.{schema}.cohorts") else 0) or 0
single.insert(1, "patient_id", range(existing_max + 1, existing_max + 1 + len(single)))

(spark.createDataFrame(single)
      .write.mode("append").option("mergeSchema", "true").saveAsTable("cohorts"))
print(f"Appended {len(single):,} single-period patients for {list(single_specs)}")

# COMMAND ----------

# MAGIC %md ### 3. Multi-period cohort for the LME study
# MAGIC `cohorts_multiperiod` is **long format**: one row per patient per period. Period 0 is
# MAGIC baseline; periods 1..K are performance periods. This is exactly the shape that does not
# MAGIC fit the single pre/post template — so it gets its own `feature_lme` + `model_lme`
# MAGIC leaves while still reusing `matching_standard`.

# COMMAND ----------

lme_study = "M_LME_004"
n_per_arm = 700
n_periods = 3            # number of performance periods after baseline
true_effect = -380.0     # per-period treatment effect (cost reduction)
missing_rate = 0.03

def make_lme_arm(treated):
    shift = 0.6 if treated else 0.0
    age = rng.normal(58 + 4 * shift, 12, n_per_arm)
    risk_score = rng.normal(1.5 + 0.5 * shift, 0.6, n_per_arm).clip(0.1, None)
    prior_cost = rng.normal(8000 + 2500 * shift, 3000, n_per_arm).clip(0, None)
    baseline = 5000 + 0.4 * prior_cost + 600 * risk_score + rng.normal(0, 1500, n_per_arm)
    rows = []
    pid_base = rng.integers(10_000_000, 90_000_000)
    for i in range(n_per_arm):
        pid = int(pid_base + i)
        # period 0 = baseline outcome
        rows.append((lme_study, pid, int(treated), float(age[i]), float(risk_score[i]),
                     float(prior_cost[i]), 0, float(baseline[i])))
        # performance periods accumulate the treatment effect (growing over time for treated)
        for p in range(1, n_periods + 1):
            eff = (true_effect * p) if treated else 0.0
            outcome = baseline[i] - 150 * p + eff + rng.normal(0, 1500)
            rows.append((lme_study, pid, int(treated), float(age[i]), float(risk_score[i]),
                         float(prior_cost[i]), p, float(outcome)))
    return rows

lme_rows = make_lme_arm(True) + make_lme_arm(False)
lme = pd.DataFrame(lme_rows, columns=[
    "study_id", "patient_id", "treatment", "age", "risk_score", "prior_cost", "period", "outcome"])

# inject a little missingness on a baseline covariate
mmask = rng.random(len(lme)) < missing_rate
lme.loc[mmask, "risk_score"] = np.nan

(spark.createDataFrame(lme)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("cohorts_multiperiod"))
print(f"Wrote {len(lme):,} rows to cohorts_multiperiod "
      f"({n_per_arm} patients/arm x {n_periods + 1} periods x 2 arms)")
display(spark.sql(f"""
  SELECT treatment, period, count(*) n, round(avg(outcome),0) avg_outcome
  FROM cohorts_multiperiod GROUP BY treatment, period ORDER BY treatment, period
"""))

# COMMAND ----------

# MAGIC %md ### Done
# MAGIC `study_config_modular`, `cohorts` (appended), and `cohorts_multiperiod` are ready.
# MAGIC Next: run **Eval Engine - MODULAR**.

# COMMAND ----------

dbutils.notebook.exit("modular_setup_complete")
