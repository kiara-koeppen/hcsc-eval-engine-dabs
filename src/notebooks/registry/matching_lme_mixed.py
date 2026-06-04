# Databricks notebook source
# MAGIC %md
# MAGIC # LME-MIXED workstream: matching_lme_mixed
# MAGIC
# MAGIC Propensity-score nearest-neighbor matching on baseline covariates for the mixed-effects
# MAGIC LME study. Reads `feat_<study_id>`, writes `matched_<study_id>` (carries `patient_id`,
# MAGIC `pre_outcome`, `weight`). Owned by the lme_mixed branch.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_LME_MIXED", "Study ID")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
print(f"[{study_id}] leaf=matching_lme_mixed")

# COMMAND ----------

import numpy as np, pandas as pd
df = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}").toPandas()
COV = ["age", "risk_score", "prior_cost"]

def propensity(frame):
    X = frame[COV].to_numpy(float); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.column_stack([np.ones(len(X)), X]); y = frame["treatment"].to_numpy(float)
    b = np.zeros(X.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-(X @ b))); W = np.clip(p * (1 - p), 1e-6, None)
        b += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return np.clip(1 / (1 + np.exp(-(X @ b))), 0.01, 0.99)

df = df.copy(); df["ps"] = propensity(df)
t = df[df.treatment == 1].copy(); c = df[df.treatment == 0].copy()
cps = c["ps"].to_numpy(); idx = [int(np.argmin(np.abs(cps - ps))) for ps in t["ps"]]
matched = pd.concat([t, c.iloc[idx].copy()], ignore_index=True); matched["weight"] = 1.0
matched = matched.drop(columns=["ps"])
n_t = int((matched.treatment == 1).sum()); n_c = int((matched.treatment == 0).sum())
print(f"[{study_id}] matched: {n_t} treated / {n_c} control")

(spark.createDataFrame(matched).write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.matched_{study_id.lower()}"))
dbutils.jobs.taskValues.set(key="n_treated", value=n_t)
