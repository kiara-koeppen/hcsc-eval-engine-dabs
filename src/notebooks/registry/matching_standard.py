# Databricks notebook source
# MAGIC %md
# MAGIC # LEAF: matching_standard
# MAGIC
# MAGIC Shared matching infrastructure: `exact` | `knn` | `ipw`, selected by the
# MAGIC `matching_method` value READ FROM THE CONFIG TABLE for this study (no widget). Reads
# MAGIC `feat_<study_id>`, writes `matched_<study_id>` with a
# MAGIC `weight` column the model stage consumes. Reused by both standard and LME studies
# MAGIC (matching operates on patient-level baseline covariates, which both produce).

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_ATT_001", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")

# Read the matching method from the config TABLE (not a widget), keyed by study_id.
matching_method = (spark.table(f"{catalog}.{schema}.{config_table}")
                        .where(f"study_id = '{study_id}'").collect()[0]["matching_method"])
print(f"[{study_id}] leaf=matching_standard method={matching_method} (read from config)")

# COMMAND ----------

import numpy as np
import pandas as pd

df = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}").toPandas()
COVARIATES = ["age", "risk_score", "prior_cost"]

# COMMAND ----------

def propensity_scores(frame: pd.DataFrame) -> np.ndarray:
    """Dependency-free logistic regression (IRLS) for propensity scores."""
    X = frame[COVARIATES].to_numpy(dtype=float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.column_stack([np.ones(len(X)), X])
    y = frame["treatment"].to_numpy(dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = np.clip(p * (1 - p), 1e-6, None)
        grad = X.T @ (y - p)
        H = X.T @ (X * W[:, None])
        beta += np.linalg.solve(H + 1e-6 * np.eye(X.shape[1]), grad)
    return np.clip(1.0 / (1.0 + np.exp(-(X @ beta))), 0.01, 0.99)

def run_matching(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    treated = frame[frame.treatment == 1].copy()
    control = frame[frame.treatment == 0].copy()

    if method == "exact":
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
        frame = frame.copy()
        frame["ps"] = propensity_scores(frame)
        t = frame[frame.treatment == 1].copy()
        c = frame[frame.treatment == 0].copy()
        c_ps = c["ps"].to_numpy()
        idx = [int(np.argmin(np.abs(c_ps - ps))) for ps in t["ps"]]
        out = pd.concat([t, c.iloc[idx].copy()], ignore_index=True)
        out["weight"] = 1.0
        return out.drop(columns=["ps"])

    if method == "ipw":
        frame = frame.copy()
        ps = propensity_scores(frame)
        frame["weight"] = np.where(frame.treatment == 1, 1.0 / ps, 1.0 / (1.0 - ps))
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

dbutils.notebook.exit(f"{n_t},{n_c}")
