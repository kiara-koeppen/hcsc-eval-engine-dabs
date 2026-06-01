# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 1 — Feature Engineering + Data-Quality Gate
# MAGIC
# MAGIC **Dispatcher notebook.** Runs the feature pipeline for ONE study, then computes a
# MAGIC data-quality score and publishes a `quality_passed` task value (`"true"`/`"false"`).
# MAGIC The job's single `condition_task` reads that value to gate the rest of the pipeline.
# MAGIC
# MAGIC `feature_method` selects the imputation strategy. Adding a new strategy = a new
# MAGIC branch in `impute()` below — **not** a new job task.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "STUDY_001", "Study ID")
dbutils.widgets.text("feature_method", "standard", "Feature method")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
feature_method = dbutils.widgets.get("feature_method")
print(f"[{study_id}] feature_method={feature_method}")

# COMMAND ----------

import pandas as pd
import numpy as np

pdf = (spark.table(f"{catalog}.{schema}.cohorts")
            .where(f"study_id = '{study_id}'")
            .toPandas())

cfg = (spark.table(f"{catalog}.{schema}.study_config")
            .where(f"study_id = '{study_id}'").collect()[0])
min_quality = float(cfg["min_quality_score"])

# COMMAND ----------

# MAGIC %md ### Imputation strategy (dispatched on `feature_method`)

# COMMAND ----------

def impute(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.copy()
    if method == "aggressive_impute":
        # Fill from within-arm median (more permissive — keeps more rows usable).
        df["risk_score"] = df.groupby("treatment")["risk_score"].transform(
            lambda s: s.fillna(s.median()))
    else:  # "standard"
        # Conservative: global median fill.
        df["risk_score"] = df["risk_score"].fillna(df["risk_score"].median())
    # Add to taste: more branches here for new feature methods. DAG unchanged.
    return df

# Quality is measured BEFORE imputation (how much real signal did we have?).
missing_fraction = float(pdf["risk_score"].isna().mean())
n_total = len(pdf)
n_treated = int((pdf["treatment"] == 1).sum())
n_control = int((pdf["treatment"] == 0).sum())

features = impute(pdf, feature_method)

# COMMAND ----------

# MAGIC %md ### Quality score → pass/fail
# MAGIC Combines completeness and sample adequacy. Compared against the study's
# MAGIC `min_quality_score` threshold from config.

# COMMAND ----------

completeness = 1.0 - missing_fraction
size_adequacy = min(1.0, min(n_treated, n_control) / 100.0)   # want >=100 per arm
quality_score = round(0.6 * completeness + 0.4 * size_adequacy, 3)
quality_passed = quality_score >= min_quality

print(f"[{study_id}] completeness={completeness:.3f} size_adequacy={size_adequacy:.3f}")
print(f"[{study_id}] quality_score={quality_score} threshold={min_quality} -> "
      f"{'PASS' if quality_passed else 'FAIL'}")

# Persist features only when usable.
if quality_passed:
    feat_table = f"{catalog}.{schema}.feat_{study_id.lower()}"
    (spark.createDataFrame(features)
          .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(feat_table))
    print(f"[{study_id}] wrote {feat_table}")
else:
    print(f"[{study_id}] data quality below threshold — skipping downstream stages")

# COMMAND ----------

# Publish the gate decision. The condition_task compares this to "true".
dbutils.jobs.taskValues.set(key="quality_passed", value=str(quality_passed).lower())
dbutils.jobs.taskValues.set(key="quality_score", value=quality_score)
