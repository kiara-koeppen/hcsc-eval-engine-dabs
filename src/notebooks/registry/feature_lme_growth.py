# Databricks notebook source
# MAGIC %md
# MAGIC # LME-GROWTH workstream: feature_lme_growth
# MAGIC
# MAGIC Feature engineering for the **growth-curve** LME model (effect that changes over time).
# MAGIC Same multi-period source as the mixed model, but this is the growth branch's OWN notebook
# MAGIC so it can diverge (e.g., engineer slope features) without touching the mixed branch.
# MAGIC Writes patient-level `feat_<study_id>` (for matching) + long `feat_<study_id>_periods`
# MAGIC (for the slope model). Sets `quality_passed` and exits with it.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_LME_GROWTH", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id"); config_table = dbutils.widgets.get("config_table")
print(f"[{study_id}] leaf=feature_lme_growth")

# COMMAND ----------

import pandas as pd, numpy as np
long = (spark.table(f"{catalog}.{schema}.cohorts_multiperiod").where(f"study_id = '{study_id}'").toPandas())
cfg = spark.table(f"{catalog}.{schema}.{config_table}").where(f"study_id = '{study_id}'").collect()[0]
min_quality = float(cfg["min_quality_score"])
# Performance window comes from the config TABLE, not a widget (HCSC's pattern).
performance_months = int(cfg["performance_months"]) if "performance_months" in cfg.asDict() else 3
print(f"[{study_id}] performance_months from config = {performance_months}")

baseline = long[long["period"] == 0].copy()
missing_fraction = float(baseline["risk_score"].isna().mean())
baseline["risk_score"] = baseline["risk_score"].fillna(baseline["risk_score"].median())
perf = long[(long["period"] >= 1) & (long["period"] <= performance_months)].copy()
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
