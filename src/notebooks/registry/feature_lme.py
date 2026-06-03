# Databricks notebook source
# MAGIC %md
# MAGIC # LEAF: feature_lme  (structurally different — multiple performance periods)
# MAGIC
# MAGIC The LME study does not fit the single pre/post template: it has one baseline period and
# MAGIC several performance periods per patient. So it gets its **own** feature leaf. It reads
# MAGIC `cohorts_multiperiod` (long format) and produces two tables:
# MAGIC
# MAGIC | Table | Shape | Used by |
# MAGIC |-------|-------|---------|
# MAGIC | `feat_<study_id>` | patient-level (one row/patient): baseline covariates + `pre_outcome` (baseline) + `post_outcome` (mean of performance periods) | `matching_standard` (so matching is reused unchanged) |
# MAGIC | `feat_<study_id>_periods` | long (patient x performance period): `patient_id, period, outcome` | `model_lme` (the longitudinal estimator) |
# MAGIC
# MAGIC This is the key idea: **different where it must be (features, model), shared where it
# MAGIC can be (matching).** Same flat DAG; the difference is which leaf notebooks the config
# MAGIC points at, not a new branch in the orchestration graph.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_LME_004", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")
print(f"[{study_id}] leaf=feature_lme")

# COMMAND ----------

import pandas as pd
import numpy as np

long = (spark.table(f"{catalog}.{schema}.cohorts_multiperiod")
             .where(f"study_id = '{study_id}'").toPandas())
cfg = (spark.table(f"{catalog}.{schema}.{config_table}")
            .where(f"study_id = '{study_id}'").collect()[0])
min_quality = float(cfg["min_quality_score"])

# COMMAND ----------

# Patient-level covariates come from baseline (period 0). Impute the covariate.
baseline = long[long["period"] == 0].copy()
missing_fraction = float(baseline["risk_score"].isna().mean())
baseline["risk_score"] = baseline["risk_score"].fillna(baseline["risk_score"].median())

perf = long[long["period"] >= 1].copy()
perf["risk_score"] = perf["risk_score"].fillna(baseline["risk_score"].median())

n_treated = int(baseline[baseline["treatment"] == 1].shape[0])
n_control = int(baseline[baseline["treatment"] == 0].shape[0])

# COMMAND ----------

# Patient-level feature table (same columns matching_standard expects):
#   pre_outcome  = baseline outcome
#   post_outcome = mean across performance periods (a summary the matching step can use)
perf_mean = perf.groupby("patient_id")["outcome"].mean().rename("post_outcome")
feat = (baseline.rename(columns={"outcome": "pre_outcome"})
                [["study_id", "patient_id", "treatment", "age", "risk_score", "prior_cost", "pre_outcome"]]
                .merge(perf_mean, on="patient_id", how="inner"))

# COMMAND ----------

completeness = 1.0 - missing_fraction
size_adequacy = min(1.0, min(n_treated, n_control) / 100.0)
quality_score = round(0.6 * completeness + 0.4 * size_adequacy, 3)
quality_passed = quality_score >= min_quality
print(f"[{study_id}] quality_score={quality_score} threshold={min_quality} -> "
      f"{'PASS' if quality_passed else 'FAIL'}")

if quality_passed:
    feat_table = f"{catalog}.{schema}.feat_{study_id.lower()}"
    (spark.createDataFrame(feat)
          .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(feat_table))
    # Long performance-period table for the longitudinal model.
    periods_table = f"{catalog}.{schema}.feat_{study_id.lower()}_periods"
    (spark.createDataFrame(perf[["patient_id", "treatment", "period", "outcome"]])
          .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(periods_table))
    print(f"[{study_id}] wrote {feat_table} and {periods_table}")
else:
    print(f"[{study_id}] below threshold — skipping downstream")

# COMMAND ----------

dbutils.notebook.exit(str(quality_passed).lower())
