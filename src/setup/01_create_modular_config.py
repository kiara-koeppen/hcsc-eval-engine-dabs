# Databricks notebook source
# MAGIC %md
# MAGIC # Setup (Modular Track): registry config + per-LME-model data
# MAGIC
# MAGIC ADDITIVE setup for the **modular / multi-team** track. Does not touch the original
# MAGIC demo's `study_config` / `cohorts`. Creates:
# MAGIC
# MAGIC | Table | Purpose |
# MAGIC |-------|---------|
# MAGIC | `study_config_modular` | One row per study. `model_family` decides the branch: `standard` rides the shared for_each loop; each `lme_*` value is its OWN separate workstream/branch. |
# MAGIC | `cohorts` (append) | Single-period treated/control patients for the standard-family studies. |
# MAGIC | `cohorts_multiperiod` | Long-format patient x period outcomes for the LME studies (one baseline + several performance periods). |
# MAGIC
# MAGIC ### Why separate LME branches (not one loop)
# MAGIC The team has **multiple** structurally-different models (Alex: "the next path is also
# MAGIC LME, but it's completely different... we're not going to use the for each"). So each LME
# MAGIC model gets its **own branch** off the job with its **own notebook trio**
# MAGIC (`feature_* -> matching_* -> model_*`). Here we scaffold two as examples:
# MAGIC `lme_mixed` (constant per-period effect) and `lme_growth` (effect that grows over
# MAGIC periods). Rename / add more to match your real models.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
print(f"Target: {catalog}.{schema}")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md ### 1. Registry config table
# MAGIC `model_family`: `standard` (shared loop) | `lme_mixed` | `lme_growth` (each its own branch).
# MAGIC For standard rows the `*_nb` columns name the leaf the dispatchers run. For LME rows they
# MAGIC document the dedicated workstream's notebooks (the LME jobs wire those notebooks directly).

# COMMAND ----------

from pyspark.sql import Row

# study_id, study_name, vendor, model_family, feature_nb, matching_nb, matching_method, model_nb, min_quality_score
registry_rows = [
    ("M_ATT_001",    "Diabetes Care Mgmt (ATT)",     "VendorA Health",   "standard",
     "feature_standard",   "matching_standard",   "exact",      "model_att",       0.80),
    ("M_DID_002",    "CHF Remote Monitoring (DID)",   "VendorB Cardio",   "standard",
     "feature_standard",   "matching_standard",   "knn",        "model_did",       0.80),
    ("M_STRAT_003",  "Maternity Support (ATT/strat)", "VendorC Maternal", "standard",
     "feature_standard",   "matching_stratified", "stratified", "model_att",       0.75),
    ("M_LME_MIXED",  "Wellness Longitudinal (mixed)", "VendorD Wellness", "lme_mixed",
     "feature_lme_mixed",  "matching_lme_mixed",  "knn",        "model_lme_mixed", 0.80),
    ("M_LME_GROWTH", "Chronic Care Longitudinal (growth)","VendorE Chronic","lme_growth",
     "feature_lme_growth", "matching_lme_growth", "knn",        "model_lme_growth",0.80),
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

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

single_specs = {  # study_id: (n_per_arm, true_effect, missing_rate)
    "M_ATT_001":   (1200, -450.0, 0.02),
    "M_DID_002":   (900,  -1.20,  0.03),
    "M_STRAT_003": (1500, -300.0, 0.04),
}

def make_arm(study_id, n, treated, true_effect, missing_rate):
    shift = 0.6 if treated else 0.0
    age = rng.normal(58 + 4 * shift, 12, n)
    risk_score = rng.normal(1.5 + 0.5 * shift, 0.6, n).clip(0.1, None)
    prior_cost = rng.normal(8000 + 2500 * shift, 3000, n).clip(0, None)
    pre_outcome = 5000 + 0.4 * prior_cost + 600 * risk_score + rng.normal(0, 1500, n)
    post_outcome = pre_outcome - 200 + (true_effect if treated else 0.0) + rng.normal(0, 1500, n)
    df = pd.DataFrame({"study_id": study_id, "treatment": int(treated), "age": age,
                       "risk_score": risk_score, "prior_cost": prior_cost,
                       "pre_outcome": pre_outcome, "post_outcome": post_outcome})
    if missing_rate > 0:
        df.loc[rng.random(n) < missing_rate, "risk_score"] = np.nan
    return df

frames = []
for sid, (n, eff, miss) in single_specs.items():
    frames.append(make_arm(sid, n, True, eff, miss))
    frames.append(make_arm(sid, n, False, eff, miss))
single = pd.concat(frames, ignore_index=True)

existing_max = (spark.table(f"{catalog}.{schema}.cohorts").selectExpr("max(patient_id) m").collect()[0]["m"]
                if spark.catalog.tableExists(f"{catalog}.{schema}.cohorts") else 0) or 0
single.insert(1, "patient_id", range(existing_max + 1, existing_max + 1 + len(single)))
(spark.createDataFrame(single)
      .write.mode("append").option("mergeSchema", "true").saveAsTable("cohorts"))
print(f"Appended {len(single):,} single-period patients for {list(single_specs)}")

# COMMAND ----------

# MAGIC %md ### 3. Multi-period cohorts for the LME studies
# MAGIC Both are long format (patient x period). `M_LME_MIXED` has a **constant** per-period
# MAGIC treatment effect; `M_LME_GROWTH` has an effect that **grows** each period. The two LME
# MAGIC models are tuned to those signals, which is exactly why they need separate flows.

# COMMAND ----------

def make_lme_study(study_id, effect_mode, n_per_arm=700, n_periods=3, per_period_effect=-380.0, missing_rate=0.03):
    rows = []
    def arm(treated):
        shift = 0.6 if treated else 0.0
        age = rng.normal(58 + 4 * shift, 12, n_per_arm)
        risk = rng.normal(1.5 + 0.5 * shift, 0.6, n_per_arm).clip(0.1, None)
        prior = rng.normal(8000 + 2500 * shift, 3000, n_per_arm).clip(0, None)
        baseline = 5000 + 0.4 * prior + 600 * risk + rng.normal(0, 1500, n_per_arm)
        pid0 = int(rng.integers(10_000_000, 90_000_000))
        for i in range(n_per_arm):
            pid = pid0 + i
            rows.append((study_id, pid, int(treated), float(age[i]), float(risk[i]), float(prior[i]), 0, float(baseline[i])))
            for p in range(1, n_periods + 1):
                if not treated:
                    eff = 0.0
                elif effect_mode == "growth":
                    eff = per_period_effect * p          # grows each period
                else:  # "mixed" -> constant per-period level shift
                    eff = per_period_effect
                outcome = baseline[i] - 150 * p + eff + rng.normal(0, 1500)
                rows.append((study_id, pid, int(treated), float(age[i]), float(risk[i]), float(prior[i]), p, float(outcome)))
    arm(True); arm(False)
    df = pd.DataFrame(rows, columns=["study_id", "patient_id", "treatment", "age", "risk_score", "prior_cost", "period", "outcome"])
    df.loc[rng.random(len(df)) < missing_rate, "risk_score"] = np.nan
    return df

lme = pd.concat([
    make_lme_study("M_LME_MIXED",  "mixed"),
    make_lme_study("M_LME_GROWTH", "growth"),
], ignore_index=True)

(spark.createDataFrame(lme)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("cohorts_multiperiod"))
print(f"Wrote {len(lme):,} rows to cohorts_multiperiod for M_LME_MIXED + M_LME_GROWTH")
display(spark.sql("""
  SELECT study_id, treatment, period, count(*) n, round(avg(outcome),0) avg_outcome
  FROM cohorts_multiperiod GROUP BY study_id, treatment, period ORDER BY study_id, treatment, period
"""))

# COMMAND ----------

dbutils.notebook.exit("modular_setup_complete")
