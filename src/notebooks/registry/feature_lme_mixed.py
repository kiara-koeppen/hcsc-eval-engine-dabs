# Databricks notebook source
# MAGIC %md
# MAGIC # LME-MIXED workstream: feature_lme_mixed
# MAGIC
# MAGIC Feature engineering for the **mixed-effects** LME model (constant per-period effect).
# MAGIC Reads `cohorts_multiperiod` for this study, builds a patient-level table
# MAGIC `feat_<study_id>` (baseline covariates + `pre_outcome` = baseline, `post_outcome` = mean
# MAGIC of performance periods) for matching, and a long `feat_<study_id>_periods` table for the
# MAGIC model. Sets `quality_passed` and exits with it.
# MAGIC
# MAGIC This notebook belongs to the **lme_mixed** branch only. It is wired directly into
# MAGIC `run_lme_mixed` (no dispatcher, no for_each).

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_LME_MIXED", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id"); config_table = dbutils.widgets.get("config_table")
print(f"[{study_id}] leaf=feature_lme_mixed")

# COMMAND ----------

import pandas as pd, numpy as np
long = (spark.table(f"{catalog}.{schema}.cohorts_multiperiod").where(f"study_id = '{study_id}'").toPandas())
cfg = spark.table(f"{catalog}.{schema}.{config_table}").where(f"study_id = '{study_id}'").collect()[0]
min_quality = float(cfg["min_quality_score"])

baseline = long[long["period"] == 0].copy()
missing_fraction = float(baseline["risk_score"].isna().mean())
baseline["risk_score"] = baseline["risk_score"].fillna(baseline["risk_score"].median())
perf = long[long["period"] >= 1].copy()
perf["risk_score"] = perf["risk_score"].fillna(baseline["risk_score"].median())
n_t = int(baseline[baseline.treatment == 1].shape[0]); n_c = int(baseline[baseline.treatment == 0].shape[0])

perf_mean = perf.groupby("patient_id")["outcome"].mean().rename("post_outcome")
feat = (baseline.rename(columns={"outcome": "pre_outcome"})
        [["study_id", "patient_id", "treatment", "age", "risk_score", "prior_cost", "pre_outcome"]]
        .merge(perf_mean, on="patient_id", how="inner"))

completeness = 1.0 - missing_fraction
size_adequacy = min(1.0, min(n_t, n_c) / 100.0)
quality_score = round(0.6 * completeness + 0.4 * size_adequacy, 3)
quality_passed = quality_score >= min_quality
print(f"[{study_id}] quality_score={quality_score} threshold={min_quality} -> {'PASS' if quality_passed else 'FAIL'}")

if quality_passed:
    (spark.createDataFrame(feat).write.mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.{schema}.feat_{study_id.lower()}"))
    (spark.createDataFrame(perf[["patient_id", "treatment", "period", "outcome"]])
        .write.mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.{schema}.feat_{study_id.lower()}_periods"))
    print(f"[{study_id}] wrote feat + periods")

dbutils.jobs.taskValues.set(key="quality_passed", value=str(quality_passed).lower())
dbutils.notebook.exit(str(quality_passed).lower())
