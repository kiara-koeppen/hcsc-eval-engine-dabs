# Databricks notebook source
# MAGIC %md
# MAGIC # Setup: Study Config + Synthetic Cohorts
# MAGIC
# MAGIC One-time setup for the **Automated Evaluation Engine** demo. Creates:
# MAGIC
# MAGIC | Table | Purpose |
# MAGIC |-------|---------|
# MAGIC | `study_config` | One row per program evaluation. The **3 key columns** (`feature_method`, `matching_method`, `model_method`) drive which logic runs — exactly the config-driven pattern the team described. |
# MAGIC | `cohorts` | Patient-level treatment/control data for every study (synthetic). |
# MAGIC
# MAGIC The downstream pipeline reads `study_config`, fans out over the rows with a
# MAGIC `for_each` task, and runs the 3-stage evaluation per study.
# MAGIC
# MAGIC One study (`STUDY_006`) is intentionally given **bad data** so you can see the
# MAGIC data-quality gate short-circuit that study while the others proceed.

# COMMAND ----------

# MAGIC %md ### Parameters (always parameterize catalog/schema — never hardcode)

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
print(f"Target: {catalog}.{schema}")

# Catalog is expected to already exist (managed UC catalog). We only ensure the
# schema. NOTE: on metastores with Default Storage, `CREATE CATALOG IF NOT EXISTS`
# can error on storage-root resolution even when the catalog exists — so we don't
# create the catalog here.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md ### 1. Study config table
# MAGIC Each row is a program evaluation. Note how `feature_method` / `matching_method` /
# MAGIC `model_method` are just **values** here — the pipeline dispatches on them. Adding a
# MAGIC new study is a one-row INSERT; it never touches the orchestration DAG.

# COMMAND ----------

from pyspark.sql import Row

config_rows = [
    # study_id,    study_name,                vendor,            feature_method,      matching, model,  min_quality_score
    ("STUDY_001", "Diabetes Care Management", "VendorA Health",  "standard",          "exact",  "att",  0.80),
    ("STUDY_002", "CHF Remote Monitoring",    "VendorB Cardio",  "standard",          "knn",    "did",  0.80),
    ("STUDY_003", "Maternity Support Program","VendorC Maternal","aggressive_impute", "ipw",    "ate",  0.75),
    ("STUDY_004", "Behavioral Health Coaching","VendorD Mind",   "standard",          "knn",    "att",  0.80),
    ("STUDY_005", "Oncology Care Navigation", "VendorE Onc",     "aggressive_impute", "exact",  "did",  0.78),
    ("STUDY_006", "Wellness Pilot (sparse)",  "VendorF Wellness","standard",          "exact",  "att",  0.85),  # will FAIL gate
]

config_df = spark.createDataFrame(
    [Row(study_id=r[0], study_name=r[1], vendor=r[2], feature_method=r[3],
         matching_method=r[4], model_method=r[5], min_quality_score=float(r[6]))
     for r in config_rows]
)
config_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("study_config")
display(spark.table("study_config"))

# COMMAND ----------

# MAGIC %md ### 2. Synthetic patient-level cohorts
# MAGIC For each study we generate treated + control patients with covariates
# MAGIC (`age`, `risk_score`, `prior_cost`) and a pre/post outcome. We bake in a known
# MAGIC **true treatment effect** per study so the estimators have something real to find.
# MAGIC `STUDY_006` gets a tiny sample + heavy missingness to trip the quality gate.

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# (study_id, n_per_arm, true_effect, missing_rate)
specs = {
    "STUDY_001": (1200, -450.0, 0.02),
    "STUDY_002": (900,  -1.20,  0.03),
    "STUDY_003": (1500, -300.0, 0.04),
    "STUDY_004": (800,  -0.80,  0.02),
    "STUDY_005": (1000, -620.0, 0.05),
    "STUDY_006": (40,    0.0,    0.45),   # tiny n + 45% missing -> fails gate
}

def make_arm(study_id, n, treated, true_effect, missing_rate):
    # Covariates. Treated patients are sicker on average (confounding) so naive
    # comparison is biased and matching/weighting actually matters.
    shift = 0.6 if treated else 0.0
    age = rng.normal(58 + 4 * shift, 12, n)
    risk_score = rng.normal(1.5 + 0.5 * shift, 0.6, n).clip(0.1, None)
    prior_cost = rng.normal(8000 + 2500 * shift, 3000, n).clip(0, None)

    # Pre-period outcome correlates with covariates.
    pre_outcome = 5000 + 0.4 * prior_cost + 600 * risk_score + rng.normal(0, 1500, n)
    # Post-period: natural trend + treatment effect (only for treated).
    post_outcome = pre_outcome - 200 + (true_effect if treated else 0.0) + rng.normal(0, 1500, n)

    df = pd.DataFrame({
        "study_id": study_id,
        "treatment": int(treated),
        "age": age,
        "risk_score": risk_score,
        "prior_cost": prior_cost,
        "pre_outcome": pre_outcome,
        "post_outcome": post_outcome,
    })
    # Inject missingness on a covariate to exercise the imputation + quality gate.
    if missing_rate > 0:
        mask = rng.random(n) < missing_rate
        df.loc[mask, "risk_score"] = np.nan
    return df

frames = []
for sid, (n, eff, miss) in specs.items():
    frames.append(make_arm(sid, n, treated=True,  true_effect=eff, missing_rate=miss))
    frames.append(make_arm(sid, n, treated=False, true_effect=eff, missing_rate=miss))

cohorts = pd.concat(frames, ignore_index=True)
cohorts.insert(1, "patient_id", range(1, len(cohorts) + 1))

print(f"Total patients: {len(cohorts):,} across {cohorts['study_id'].nunique()} studies")

(spark.createDataFrame(cohorts)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("cohorts"))

display(spark.sql("""
  SELECT study_id, treatment, count(*) AS n,
         round(avg(risk_score),2) AS avg_risk,
         round(avg(post_outcome - pre_outcome),1) AS avg_delta
  FROM cohorts GROUP BY study_id, treatment ORDER BY study_id, treatment
"""))

# COMMAND ----------

# MAGIC %md ### Done
# MAGIC `study_config` and `cohorts` are ready. Next: run **Eval Engine - AFTER** (or the
# MAGIC single-study / before jobs).

# COMMAND ----------

dbutils.notebook.exit("setup_complete")
