# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: Matching (Conditional)
# MAGIC
# MAGIC Runs only when the model needs it (Phase 1 routed here). Reads `feat_<study_id>`,
# MAGIC balances treatment/control with the configured method, writes `matched_<study_id>`,
# MAGIC then sets nextphase to the configured model notebook.

# COMMAND ----------

# MAGIC %run ./common_nb

# COMMAND ----------

import time, numpy as np, pandas as pd
t0 = time.time()
method = pe_config["matching_method"]
df = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}").toPandas()
COV = ["age", "risk_score", "prior_cost"]

def propensity(frame):
    X = frame[COV].to_numpy(float); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.column_stack([np.ones(len(X)), X]); y = frame["treatment"].to_numpy(float); b = np.zeros(X.shape[1])
    for _ in range(25):
        p = 1 / (1 + np.exp(-(X @ b))); W = np.clip(p * (1 - p), 1e-6, None)
        b += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return np.clip(1 / (1 + np.exp(-(X @ b))), 0.01, 0.99)

if method == "exact":
    for d in (df,):
        df["age_b"] = pd.cut(df["age"], [0, 50, 60, 70, 200], labels=False)
        df["risk_b"] = pd.cut(df["risk_score"], [0, 1, 1.5, 2, 100], labels=False)
    t = df[df.treatment == 1]; c = df[df.treatment == 0]
    strata = set(map(tuple, t[["age_b", "risk_b"]].dropna().values)) & set(map(tuple, c[["age_b", "risk_b"]].dropna().values))
    keep = lambda d: d[[tuple(x) in strata for x in d[["age_b", "risk_b"]].values]]
    matched = pd.concat([keep(t), keep(c)], ignore_index=True).drop(columns=["age_b", "risk_b"]); matched["weight"] = 1.0
else:  # knn (default)
    df = df.copy(); df["ps"] = propensity(df)
    t = df[df.treatment == 1].copy(); c = df[df.treatment == 0].copy()
    idx = [int(np.argmin(np.abs(c["ps"].to_numpy() - ps))) for ps in t["ps"]]
    matched = pd.concat([t, c.iloc[idx]], ignore_index=True).drop(columns=["ps"]); matched["weight"] = 1.0

n_t = int((matched.treatment == 1).sum()); n_c = int((matched.treatment == 0).sum())
spark.createDataFrame(matched).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog}.{schema}.matched_{study_id.lower()}")

log_step("2", "Matching", "PASS",
         pass_description=f"method={method}; matched {n_t} treated / {n_c} control",
         n_input=len(df), n_treated=n_t, execution_time=time.time() - t0)

dbutils.notebook.exit(f"model_{pe_config['model_type']}")
