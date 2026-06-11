# Databricks notebook source
# MAGIC %md
# MAGIC # Setup (PE pipeline): config table + cohorts
# MAGIC
# MAGIC Builds the config + data for the `pe_pipeline` track, the faithful reference of HCSC's
# MAGIC documented architecture. Creates:
# MAGIC
# MAGIC | Table | Purpose |
# MAGIC |-------|---------|
# MAGIC | `pe_study_config` | One row per study/job. Drives everything (model_type, whether matching is needed, matching method, fallback model, periods). |
# MAGIC | `pe_cohorts` | Single-period treated/control patients per study. The Wilcoxon study is given **skewed** pre/post differences so its symmetry assumption fails and it falls back to the Sign Test. |

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}"); spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md ### 1. pe_study_config (the centralized config table)
# MAGIC `needs_matching` makes Phase 2 conditional (HCSC: "Matching runs only when required by
# MAGIC the model"). `fallback_model` is the next notebook to route to if a model's assumption
# MAGIC fails (HCSC: "Wilcoxon -> Sign Test on failure"). `experiment_name` blank -> common_nb
# MAGIC defaults to a per-user MLflow experiment.

# COMMAND ----------

from pyspark.sql import Row

# study_id, study_name, model_type, needs_matching, matching_method, fallback_model, baseline_months, performance_months, min_quality_score, experiment_name
rows = [
    ("PE_ATT_001",      "Diabetes ATT (6M perf)",   "att",      True,  "exact", "None",            12, 6, 0.80, ""),
    ("PE_DID_002",      "CHF DiD (6M perf)",         "did",      True,  "knn",   "None",            12, 6, 0.80, ""),
    ("PE_WILCOXON_003", "Wellness pre/post (Wilcoxon)","wilcoxon", False, "",      "model_sign_test", 12, 6, 0.75, ""),
]
cfg = spark.createDataFrame([
    Row(study_id=r[0], study_name=r[1], model_type=r[2], needs_matching=bool(r[3]),
        matching_method=r[4], fallback_model=r[5], baseline_months=int(r[6]),
        performance_months=int(r[7]), min_quality_score=float(r[8]), experiment_name=r[9])
    for r in rows
])
cfg.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("pe_study_config")
display(spark.table("pe_study_config"))

# COMMAND ----------

# MAGIC %md ### 2. pe_cohorts (PE_WILCOXON_003 gets skewed pre/post differences)

# COMMAND ----------

import numpy as np, pandas as pd
rng = np.random.default_rng(11)

def make_arm(study_id, n, treated, true_effect, missing_rate, skew=False):
    shift = 0.6 if treated else 0.0
    age = rng.normal(58 + 4 * shift, 12, n)
    risk = rng.normal(1.5 + 0.5 * shift, 0.6, n).clip(0.1, None)
    prior = rng.normal(8000 + 2500 * shift, 3000, n).clip(0, None)
    pre = 5000 + 0.4 * prior + 600 * risk + rng.normal(0, 1500, n)
    if skew and treated:
        # Asymmetric (right-skewed) treatment differences -> violates Wilcoxon symmetry
        # assumption -> the model should fall back to the Sign Test.
        diff = true_effect + (rng.exponential(900, n) - 900) * 3.0
    else:
        diff = (true_effect if treated else 0.0) + rng.normal(0, 1500, n)
    post = pre - 200 + diff
    df = pd.DataFrame({"study_id": study_id, "treatment": int(treated), "age": age,
                       "risk_score": risk, "prior_cost": prior, "pre_outcome": pre, "post_outcome": post})
    if missing_rate > 0:
        df.loc[rng.random(n) < missing_rate, "risk_score"] = np.nan
    return df

specs = {  # study_id: (n_per_arm, true_effect, missing_rate, skew)
    "PE_ATT_001":      (1200, -450.0, 0.02, False),
    "PE_DID_002":      (900,  -1.20,  0.03, False),
    "PE_WILCOXON_003": (800,  -350.0, 0.02, True),
}
frames = []
for sid, (n, eff, miss, sk) in specs.items():
    frames.append(make_arm(sid, n, True,  eff, miss, skew=sk))
    frames.append(make_arm(sid, n, False, eff, miss, skew=False))
coh = pd.concat(frames, ignore_index=True)
coh.insert(1, "patient_id", range(1, len(coh) + 1))
(spark.createDataFrame(coh)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("pe_cohorts"))
print(f"Wrote {len(coh):,} rows to pe_cohorts for {list(specs)}")

# COMMAND ----------

dbutils.notebook.exit("pe_setup_complete")
