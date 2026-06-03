# Databricks notebook source
# MAGIC %md
# MAGIC # LEAF: model_lme  (owned by the longitudinal team — structurally different)
# MAGIC
# MAGIC The longitudinal model consumes the multi-period outputs of `feature_lme`:
# MAGIC `matched_<study_id>` (patient-level, with `weight` + baseline `pre_outcome`) joined to
# MAGIC `feat_<study_id>_periods` (one row per patient per performance period).
# MAGIC
# MAGIC Estimate: a longitudinal difference-in-differences. For each matched patient we take the
# MAGIC mean performance-period outcome minus their baseline, then compare the weighted treated
# MAGIC change to the weighted control change. CI via a patient-level (cluster) bootstrap.
# MAGIC Appends one row to `evaluation_results` with `model_method = "lme"`, so it lands in the
# MAGIC same results table and dashboard as every other study.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_LME_004", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")
print(f"[{study_id}] leaf=model_lme")

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pyspark.sql import Row

matched = spark.table(f"{catalog}.{schema}.matched_{study_id.lower()}").toPandas()
periods = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}_periods").toPandas()
cfg = (spark.table(f"{catalog}.{schema}.{config_table}")
            .where(f"study_id='{study_id}'").collect()[0])
matching_method = cfg["matching_method"]

# Per-patient mean performance-period outcome.
perf_mean = periods.groupby("patient_id")["outcome"].mean().rename("perf_mean")

# Join onto the matched cohort (keeps matching weights; duplicates from knn reuse are
# preserved intentionally, each carrying its patient's longitudinal change).
panel = matched.merge(perf_mean, on="patient_id", how="inner")
panel["delta"] = panel["perf_mean"] - panel["pre_outcome"]   # change vs baseline

def wmean(s, w):
    return float(np.average(np.asarray(s, float), weights=np.asarray(w, float)))

def lme_effect(frame):
    t = frame[frame.treatment == 1]
    c = frame[frame.treatment == 0]
    return wmean(t.delta, t.weight) - wmean(c.delta, c.weight)

point = lme_effect(panel)

# Cluster bootstrap over matched rows.
rng = np.random.default_rng(0)
boot = []
for _ in range(300):
    samp = panel.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9)))
    if (samp.treatment == 1).any() and (samp.treatment == 0).any():
        boot.append(lme_effect(samp))
std_error = float(np.std(boot)) if boot else 0.0
ci_low, ci_high = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (point, point)

# Naive: unadjusted performance-period mean difference (no baseline, no matching).
raw = spark.table(f"{catalog}.{schema}.cohorts_multiperiod").where(
    f"study_id='{study_id}' AND period >= 1").toPandas()
naive = float(raw[raw.treatment == 1].outcome.mean() - raw[raw.treatment == 0].outcome.mean())
print(f"[{study_id}] lme estimate={point:,.2f} CI[{ci_low:,.1f},{ci_high:,.1f}] naive={naive:,.2f}")

# COMMAND ----------

result = Row(
    study_id=study_id, run_ts=datetime.now(timezone.utc).isoformat(),
    matching_method=matching_method, model_method="lme",
    n_treated=int((panel.treatment == 1).sum()), n_control=int((panel.treatment == 0).sum()),
    estimate=round(point, 3), std_error=round(std_error, 3),
    ci_low=round(ci_low, 3), ci_high=round(ci_high, 3),
    naive_estimate=round(naive, 3),
    significant=bool(ci_low > 0 or ci_high < 0), source="modular",
)
(spark.createDataFrame([result])
      .write.mode("append").option("mergeSchema", "true")
      .saveAsTable(f"{catalog}.{schema}.evaluation_results"))
print(f"[{study_id}] appended result (model_lme)")

dbutils.notebook.exit(str(round(point, 3)))
