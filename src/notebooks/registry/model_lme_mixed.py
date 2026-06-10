# Databricks notebook source
# MAGIC %md
# MAGIC # LME-MIXED workstream: model_lme_mixed
# MAGIC
# MAGIC Mixed-effects-style estimate: per matched patient, mean performance-period outcome minus
# MAGIC baseline (a level change); weighted treated change minus weighted control change. Cluster
# MAGIC bootstrap CI. Appends one row to `evaluation_results` with `model_method = "lme_mixed"`.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_LME_MIXED", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")
dbutils.widgets.text("job_run_id", "", "Job run id (reproducibility stamp)")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id"); config_table = dbutils.widgets.get("config_table")
job_run_id = dbutils.widgets.get("job_run_id")
print(f"[{study_id}] leaf=model_lme_mixed")

# COMMAND ----------

import numpy as np, pandas as pd
from datetime import datetime, timezone
from pyspark.sql import Row

matched = spark.table(f"{catalog}.{schema}.matched_{study_id.lower()}").toPandas()
periods = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}_periods").toPandas()
cfg = spark.table(f"{catalog}.{schema}.{config_table}").where(f"study_id='{study_id}'").collect()[0]

def _delta_version(tbl):
    try:
        from pyspark.sql import functions as F
        return int(spark.sql(f"DESCRIBE HISTORY {tbl}").agg(F.max("version")).collect()[0][0])
    except Exception:
        return -1
config_version = _delta_version(f"{catalog}.{schema}.{config_table}")
data_version = _delta_version(f"{catalog}.{schema}.cohorts_multiperiod")

perf_mean = periods.groupby("patient_id")["outcome"].mean().rename("perf_mean")
panel = matched.merge(perf_mean, on="patient_id", how="inner")
panel["delta"] = panel["perf_mean"] - panel["pre_outcome"]

def wmean(s, w): return float(np.average(np.asarray(s, float), weights=np.asarray(w, float)))
def effect(f):
    t = f[f.treatment == 1]; c = f[f.treatment == 0]
    return wmean(t.delta, t.weight) - wmean(c.delta, c.weight)

point = effect(panel)
rng = np.random.default_rng(0); boot = []
for _ in range(300):
    s = panel.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9)))
    if (s.treatment == 1).any() and (s.treatment == 0).any(): boot.append(effect(s))
se = float(np.std(boot)) if boot else 0.0
lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (point, point)
raw = spark.table(f"{catalog}.{schema}.cohorts_multiperiod").where(f"study_id='{study_id}' AND period>=1").toPandas()
naive = float(raw[raw.treatment == 1].outcome.mean() - raw[raw.treatment == 0].outcome.mean())
print(f"[{study_id}] lme_mixed estimate={point:,.2f} CI[{lo:,.1f},{hi:,.1f}] naive={naive:,.2f}")

# COMMAND ----------

(spark.createDataFrame([Row(
    study_id=study_id, run_ts=datetime.now(timezone.utc).isoformat(),
    matching_method=cfg["matching_method"], model_method="lme_mixed",
    n_treated=int((panel.treatment == 1).sum()), n_control=int((panel.treatment == 0).sum()),
    estimate=round(point, 3), std_error=round(se, 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
    naive_estimate=round(naive, 3), significant=bool(lo > 0 or hi < 0), source="modular",
    config_version=config_version, data_version=data_version, job_run_id=job_run_id)])
    .write.mode("append").option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.evaluation_results"))
print(f"[{study_id}] appended result (model_lme_mixed)")
dbutils.notebook.exit(str(round(point, 3)))
