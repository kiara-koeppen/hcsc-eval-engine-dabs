# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 3 — Evaluation Model (single PARAMETERIZED dispatcher)
# MAGIC
# MAGIC One notebook, every estimator. `model_method` selects the estimator:
# MAGIC `att` · `ate` · `did` · `gee` · `mixed`. Each computes a treatment-effect estimate
# MAGIC on the matched/weighted cohort, with a bootstrap 95% CI. Results are appended to
# MAGIC `evaluation_results`.
# MAGIC
# MAGIC > 20 models? Add 20 branches in `estimate()`. The DAG never grows. In the *before*
# MAGIC > job each of these is a separate task, duplicated under every matching branch.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "STUDY_001", "Study ID")
dbutils.widgets.text("model_method", "att", "Model method")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
model_method = dbutils.widgets.get("model_method")
print(f"[{study_id}] model_method={model_method}")

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timezone

matched = spark.table(f"{catalog}.{schema}.matched_{study_id.lower()}").toPandas()
cfg = spark.table(f"{catalog}.{schema}.study_config").where(f"study_id='{study_id}'").collect()[0]
matching_method = cfg["matching_method"]
COVARIATES = ["age", "risk_score", "prior_cost"]

# COMMAND ----------

# MAGIC %md ### Estimators (dispatched on `model_method`)

# COMMAND ----------

def wmean(s, w):
    return float(np.average(np.asarray(s, float), weights=np.asarray(w, float)))

def wls_treatment_coef(frame, extra_cols):
    """Weighted least squares; returns the coefficient on `treatment`."""
    cols = ["treatment"] + extra_cols
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy(float) for c in cols])
    y = frame["post_outcome"].to_numpy(float)
    w = frame["weight"].to_numpy(float)
    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return float(beta[1])  # index 1 == treatment

def estimate(frame, method):
    t = frame[frame.treatment == 1]
    c = frame[frame.treatment == 0]
    if method in ("att", "ate"):
        return wmean(t.post_outcome, t.weight) - wmean(c.post_outcome, c.weight)
    if method == "did":
        dt = t.post_outcome - t.pre_outcome
        dc = c.post_outcome - c.pre_outcome
        return wmean(dt, t.weight) - wmean(dc, c.weight)
    if method == "gee":                       # population-average, covariate-adjusted
        return wls_treatment_coef(frame, COVARIATES)
    if method == "mixed":                     # ANCOVA-style: adjust for baseline too
        return wls_treatment_coef(frame, ["pre_outcome"] + COVARIATES)
    raise ValueError(f"Unknown model_method: {method}")

# COMMAND ----------

# MAGIC %md ### Point estimate + bootstrap 95% CI

# COMMAND ----------

point = estimate(matched, model_method)

rng = np.random.default_rng(7)
t_idx = matched.index[matched.treatment == 1].to_numpy()
c_idx = matched.index[matched.treatment == 0].to_numpy()
boot = []
for _ in range(300):
    samp = matched.loc[np.concatenate([
        rng.choice(t_idx, len(t_idx), replace=True),
        rng.choice(c_idx, len(c_idx), replace=True),
    ])]
    try:
        boot.append(estimate(samp, model_method))
    except Exception:
        pass
boot = np.array(boot)
std_error = float(boot.std())
ci_low, ci_high = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

# Naive (unadjusted, unmatched) estimate for contrast — shows what matching corrected.
raw = spark.table(f"{catalog}.{schema}.cohorts").where(f"study_id='{study_id}'").toPandas()
naive = float(raw[raw.treatment == 1].post_outcome.mean() - raw[raw.treatment == 0].post_outcome.mean())

print(f"[{study_id}] {model_method} estimate = {point:,.2f}  95% CI [{ci_low:,.1f}, {ci_high:,.1f}]")
print(f"[{study_id}] naive (unadjusted) = {naive:,.2f}  -> matching/weighting shift = {point - naive:,.2f}")

# COMMAND ----------

# MAGIC %md ### Append to `evaluation_results`
# MAGIC Concurrent `for_each` runs each append their own row — Delta handles concurrent appends.

# COMMAND ----------

from pyspark.sql import Row
result = Row(
    study_id=study_id,
    run_ts=datetime.now(timezone.utc).isoformat(),
    matching_method=matching_method,
    model_method=model_method,
    n_treated=int((matched.treatment == 1).sum()),
    n_control=int((matched.treatment == 0).sum()),
    estimate=round(point, 3),
    std_error=round(std_error, 3),
    ci_low=round(ci_low, 3),
    ci_high=round(ci_high, 3),
    naive_estimate=round(naive, 3),
    significant=bool(ci_low > 0 or ci_high < 0),
    source="after",
)
results_table = f"{catalog}.{schema}.evaluation_results"
spark.createDataFrame([result]).write.mode("append").option("mergeSchema", "true").saveAsTable(results_table)
print(f"[{study_id}] appended result to {results_table}")

dbutils.jobs.taskValues.set(key="estimate", value=round(point, 3))
dbutils.jobs.taskValues.set(key="significant", value=str(bool(ci_low > 0 or ci_high < 0)).lower())
