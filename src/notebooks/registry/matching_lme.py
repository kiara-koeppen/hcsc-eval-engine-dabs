# Databricks notebook source
# MAGIC %md
# MAGIC # LEAF: matching_lme  (part of the dedicated LME workstream)
# MAGIC
# MAGIC This notebook belongs to the **separate LME workstream**, not the standard `for_each`
# MAGIC flow. The LME team owns it. It reads the patient-level `feat_<study_id>` produced by
# MAGIC `feature_lme` and writes `matched_<study_id>` with a `weight` column, carrying
# MAGIC `patient_id` and `pre_outcome` through so `model_lme` can join the performance periods.
# MAGIC
# MAGIC Implementation: propensity-score nearest-neighbor matching (dependency-free numpy).
# MAGIC It is intentionally its own file so the LME path can diverge from standard matching
# MAGIC without affecting anyone else.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_LME_004", "Study ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
print(f"[{study_id}] leaf=matching_lme")

# COMMAND ----------

import numpy as np
import pandas as pd

df = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}").toPandas()
COVARIATES = ["age", "risk_score", "prior_cost"]

def propensity_scores(frame: pd.DataFrame) -> np.ndarray:
    X = frame[COVARIATES].to_numpy(dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.column_stack([np.ones(len(X)), X])
    y = frame["treatment"].to_numpy(dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = np.clip(p * (1 - p), 1e-6, None)
        beta += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return np.clip(1.0 / (1.0 + np.exp(-(X @ beta))), 0.01, 0.99)

frame = df.copy()
frame["ps"] = propensity_scores(frame)
t = frame[frame.treatment == 1].copy()
c = frame[frame.treatment == 0].copy()
c_ps = c["ps"].to_numpy()
idx = [int(np.argmin(np.abs(c_ps - ps))) for ps in t["ps"]]
matched = pd.concat([t, c.iloc[idx].copy()], ignore_index=True)
matched["weight"] = 1.0
matched = matched.drop(columns=["ps"])

n_t = int((matched.treatment == 1).sum())
n_c = int((matched.treatment == 0).sum())
print(f"[{study_id}] LME matched cohort: {n_t} treated / {n_c} control")

# COMMAND ----------

out_table = f"{catalog}.{schema}.matched_{study_id.lower()}"
(spark.createDataFrame(matched)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(out_table))
print(f"[{study_id}] wrote {out_table}")

dbutils.jobs.taskValues.set(key="n_treated", value=n_t)
dbutils.jobs.taskValues.set(key="n_control", value=n_c)
