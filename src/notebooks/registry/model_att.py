# Databricks notebook source
# MAGIC %md
# MAGIC # LEAF: model_att  (owned by the ATT team)
# MAGIC
# MAGIC One model, one notebook, one owner. Estimates the Average Treatment effect on the
# MAGIC Treated as the weighted mean outcome difference on the matched cohort, with a bootstrap
# MAGIC 95% CI, and appends one row to `evaluation_results`. A different team's model is a
# MAGIC different notebook (`model_did`, `model_lme`, ...) — they never touch this file.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_ATT_001", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")
print(f"[{study_id}] leaf=model_att")

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pyspark.sql import Row

matched = spark.table(f"{catalog}.{schema}.matched_{study_id.lower()}").toPandas()
cfg = (spark.table(f"{catalog}.{schema}.{config_table}")
            .where(f"study_id='{study_id}'").collect()[0])
matching_method = cfg["matching_method"]

def wmean(s, w):
    return float(np.average(np.asarray(s, float), weights=np.asarray(w, float)))

def att(frame):
    t = frame[frame.treatment == 1]
    c = frame[frame.treatment == 0]
    return wmean(t.post_outcome, t.weight) - wmean(c.post_outcome, c.weight)

point = att(matched)

# Bootstrap CI over rows.
rng = np.random.default_rng(0)
boot = []
for _ in range(300):
    samp = matched.sample(frac=1.0, replace=True, random_state=int(rng.integers(1e9)))
    if (samp.treatment == 1).any() and (samp.treatment == 0).any():
        boot.append(att(samp))
std_error = float(np.std(boot)) if boot else 0.0
ci_low, ci_high = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (point, point)

# Naive (unadjusted) contrast from raw cohorts.
raw = spark.table(f"{catalog}.{schema}.cohorts").where(f"study_id='{study_id}'").toPandas()
naive = float(raw[raw.treatment == 1].post_outcome.mean() - raw[raw.treatment == 0].post_outcome.mean())
print(f"[{study_id}] att estimate={point:,.2f} CI[{ci_low:,.1f},{ci_high:,.1f}] naive={naive:,.2f}")

# COMMAND ----------

result = Row(
    study_id=study_id, run_ts=datetime.now(timezone.utc).isoformat(),
    matching_method=matching_method, model_method="att",
    n_treated=int((matched.treatment == 1).sum()), n_control=int((matched.treatment == 0).sum()),
    estimate=round(point, 3), std_error=round(std_error, 3),
    ci_low=round(ci_low, 3), ci_high=round(ci_high, 3),
    naive_estimate=round(naive, 3),
    significant=bool(ci_low > 0 or ci_high < 0), source="modular",
)
(spark.createDataFrame([result])
      .write.mode("append").option("mergeSchema", "true")
      .saveAsTable(f"{catalog}.{schema}.evaluation_results"))
print(f"[{study_id}] appended result (model_att)")

dbutils.notebook.exit(str(round(point, 3)))
