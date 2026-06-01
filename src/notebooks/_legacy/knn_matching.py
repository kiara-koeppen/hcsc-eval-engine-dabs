# Databricks notebook source
# MAGIC %md
# MAGIC # [LEGACY] knn_matching
# MAGIC Another standalone matching notebook — its own task, reached only when `route_knn`
# MAGIC is true. The duplication of orchestration scaffolding per method is the cost.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "STUDY_001", "Study ID")
catalog, schema, study_id = (dbutils.widgets.get(x) for x in ("catalog", "schema", "study_id"))

# COMMAND ----------

import numpy as np, pandas as pd
df = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}").toPandas()
COV = ["age", "risk_score", "prior_cost"]

X = df[COV].to_numpy(float); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
X = np.column_stack([np.ones(len(X)), X]); y = df["treatment"].to_numpy(float)
beta = np.zeros(X.shape[1])
for _ in range(25):
    p = 1 / (1 + np.exp(-(X @ beta))); W = np.clip(p * (1 - p), 1e-6, None)
    beta += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
df["ps"] = np.clip(1 / (1 + np.exp(-(X @ beta))), 0.01, 0.99)

t, c = df[df.treatment == 1].copy(), df[df.treatment == 0].copy()
c_ps = c["ps"].to_numpy()
idx = [int(np.argmin(np.abs(c_ps - ps))) for ps in t["ps"]]
out = pd.concat([t, c.iloc[idx].copy()], ignore_index=True).drop(columns=["ps"])
out["weight"] = 1.0
(spark.createDataFrame(out).write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{catalog}.{schema}.matched_{study_id.lower()}"))
print(f"[{study_id}] knn matching -> {int((out.treatment==1).sum())}/{int((out.treatment==0).sum())}")
