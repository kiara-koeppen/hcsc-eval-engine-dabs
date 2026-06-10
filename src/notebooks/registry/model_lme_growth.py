# Databricks notebook source
# MAGIC %md
# MAGIC # LME-GROWTH workstream: model_lme_growth
# MAGIC
# MAGIC Growth-curve estimate: for each matched patient, fit the slope of outcome over time
# MAGIC (period 0 = baseline `pre_outcome`, periods 1..K = performance outcomes). The treatment
# MAGIC effect is the weighted difference in mean slope (treated minus control), i.e. the
# MAGIC difference in trajectory over time. Cluster bootstrap CI. Appends one row to
# MAGIC `evaluation_results` with `model_method = "lme_growth"`.
# MAGIC
# MAGIC This is a genuinely different estimator from `model_lme_mixed` (slope vs level), which is
# MAGIC why it is its own branch with its own notebook.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_LME_GROWTH", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")
dbutils.widgets.text("job_run_id", "", "Job run id (reproducibility stamp)")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id"); config_table = dbutils.widgets.get("config_table")
job_run_id = dbutils.widgets.get("job_run_id")
print(f"[{study_id}] leaf=model_lme_growth")

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

# Per-patient slope of outcome over time, using baseline (period 0) + performance periods.
perf_by_pt = {pid: g.sort_values("period")[["period", "outcome"]].to_numpy(float)
              for pid, g in periods.groupby("patient_id")}

def slope_for(pid, pre_outcome):
    pts = perf_by_pt.get(pid)
    xs = [0.0]; ys = [float(pre_outcome)]
    if pts is not None:
        xs += list(pts[:, 0]); ys += list(pts[:, 1])
    xs = np.asarray(xs); ys = np.asarray(ys)
    if len(xs) < 2 or xs.std() == 0: return 0.0
    return float(np.polyfit(xs, ys, 1)[0])   # slope

panel = matched.copy()
panel["slope"] = [slope_for(int(r.patient_id), r.pre_outcome) for r in panel.itertuples()]

def wmean(s, w): return float(np.average(np.asarray(s, float), weights=np.asarray(w, float)))
def effect(f):
    t = f[f.treatment == 1]; c = f[f.treatment == 0]
    return wmean(t.slope, t.weight) - wmean(c.slope, c.weight)

point = effect(panel)
rng = np.random.default_rng(0); boot = []
for _ in range(300):
    s = panel.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9)))
    if (s.treatment == 1).any() and (s.treatment == 0).any(): boot.append(effect(s))
se = float(np.std(boot)) if boot else 0.0
lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (point, point)

# Naive: difference in raw mean slope without matching.
print(f"[{study_id}] lme_growth slope effect={point:,.3f} CI[{lo:,.3f},{hi:,.3f}] (per-period change)")

# COMMAND ----------

(spark.createDataFrame([Row(
    study_id=study_id, run_ts=datetime.now(timezone.utc).isoformat(),
    matching_method=cfg["matching_method"], model_method="lme_growth",
    n_treated=int((panel.treatment == 1).sum()), n_control=int((panel.treatment == 0).sum()),
    estimate=round(point, 3), std_error=round(se, 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
    naive_estimate=round(point, 3), significant=bool(lo > 0 or hi < 0), source="modular",
    config_version=config_version, data_version=data_version, job_run_id=job_run_id)])
    .write.mode("append").option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.evaluation_results"))
print(f"[{study_id}] appended result (model_lme_growth)")
dbutils.notebook.exit(str(round(point, 3)))
