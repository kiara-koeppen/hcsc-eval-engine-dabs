# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 2 — Matching (single PARAMETERIZED dispatcher)
# MAGIC
# MAGIC The crux of the scalable pattern: **one** notebook handles every matching method.
# MAGIC `matching_method` (`exact` | `knn` | `ipw`) selects the algorithm inside `run_matching()`.
# MAGIC
# MAGIC > Add a 4th, 5th … 10th method by adding a branch here. The job DAG stays
# MAGIC > identical. Contrast with the *before* job, where each method is its own task
# MAGIC > gated by its own condition.
# MAGIC
# MAGIC Output: a `matched_<study_id>` table with a `weight` column the model stage consumes.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "STUDY_001", "Study ID")
dbutils.widgets.text("matching_method", "exact", "Matching method")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
matching_method = dbutils.widgets.get("matching_method")
print(f"[{study_id}] matching_method={matching_method}")

# COMMAND ----------

import numpy as np
import pandas as pd

df = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}").toPandas()
COVARIATES = ["age", "risk_score", "prior_cost"]

# COMMAND ----------

# MAGIC %md ### Helper: propensity scores via lightweight logistic regression (IRLS)
# MAGIC Dependency-free (numpy only) so it runs anywhere on serverless.

# COMMAND ----------

def propensity_scores(frame: pd.DataFrame) -> np.ndarray:
    X = frame[COVARIATES].to_numpy(dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)            # standardize
    X = np.column_stack([np.ones(len(X)), X])          # intercept
    y = frame["treatment"].to_numpy(dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(25):                                # IRLS / Newton steps
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-6, None)
        grad = X.T @ (y - p)
        H = X.T @ (X * W[:, None])
        beta += np.linalg.solve(H + 1e-6 * np.eye(X.shape[1]), grad)
    return np.clip(1.0 / (1.0 + np.exp(-(X @ beta))), 0.01, 0.99)

# COMMAND ----------

# MAGIC %md ### Dispatch on `matching_method`

# COMMAND ----------

def run_matching(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    treated = frame[frame.treatment == 1].copy()
    control = frame[frame.treatment == 0].copy()

    if method == "exact":
        # Coarsened exact matching on age & risk buckets; keep arms present in
        # the same stratum, weight 1.0.
        for d in (treated, control):
            d["age_b"] = pd.cut(d["age"], bins=[0, 50, 60, 70, 200], labels=False)
            d["risk_b"] = pd.cut(d["risk_score"], bins=[0, 1, 1.5, 2, 100], labels=False)
        strata = set(map(tuple, treated[["age_b", "risk_b"]].dropna().values)) & \
                 set(map(tuple, control[["age_b", "risk_b"]].dropna().values))
        keep = lambda d: d[[tuple(x) in strata for x in d[["age_b", "risk_b"]].values]]
        out = pd.concat([keep(treated), keep(control)], ignore_index=True)
        out["weight"] = 1.0
        return out.drop(columns=["age_b", "risk_b"])

    if method == "knn":
        # 1:1 nearest-neighbor on propensity score (greedy, with replacement).
        frame = frame.copy()
        frame["ps"] = propensity_scores(frame)
        t = frame[frame.treatment == 1].copy()
        c = frame[frame.treatment == 0].copy()
        c_ps = c["ps"].to_numpy()
        idx = [int(np.argmin(np.abs(c_ps - ps))) for ps in t["ps"]]
        matched_controls = c.iloc[idx].copy()
        out = pd.concat([t, matched_controls], ignore_index=True)
        out["weight"] = 1.0
        return out.drop(columns=["ps"])

    if method == "ipw":
        # Inverse-probability weighting: keep everyone, weight by 1/P(assignment).
        frame = frame.copy()
        ps = propensity_scores(frame)
        frame["weight"] = np.where(frame.treatment == 1, 1.0 / ps, 1.0 / (1.0 - ps))
        # stabilize + trim extreme weights
        frame["weight"] = frame["weight"].clip(upper=frame["weight"].quantile(0.99))
        return frame

    raise ValueError(f"Unknown matching_method: {method}")

matched = run_matching(df, matching_method)
n_t = int((matched.treatment == 1).sum())
n_c = int((matched.treatment == 0).sum())
print(f"[{study_id}] matched cohort: {n_t} treated / {n_c} control")

# COMMAND ----------

out_table = f"{catalog}.{schema}.matched_{study_id.lower()}"
(spark.createDataFrame(matched)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(out_table))
print(f"[{study_id}] wrote {out_table}")

dbutils.jobs.taskValues.set(key="n_treated", value=n_t)
dbutils.jobs.taskValues.set(key="n_control", value=n_c)
