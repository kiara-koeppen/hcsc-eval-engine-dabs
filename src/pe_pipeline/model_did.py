# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: model_did  (Difference-in-Differences)
# MAGIC
# MAGIC Reads `matched_<study_id>`, appends one record to `pe_evaluation_results`, exits `None`.

# COMMAND ----------

# MAGIC %run ./common_nb

# COMMAND ----------

import time, numpy as np, pandas as pd
from pyspark.sql import Row
from datetime import datetime, timezone
t0 = time.time()

m = spark.table(f"{catalog}.{schema}.matched_{study_id.lower()}").toPandas()
def wmean(s, w): return float(np.average(np.asarray(s, float), weights=np.asarray(w, float)))
def did(f):
    t = f[f.treatment == 1]; c = f[f.treatment == 0]
    return wmean(t.post_outcome - t.pre_outcome, t.weight) - wmean(c.post_outcome - c.pre_outcome, c.weight)
point = did(m)
rng = np.random.default_rng(0); boot = [did(m.sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))) for _ in range(300)]
se = float(np.std(boot)); lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
n_t = int((m.treatment == 1).sum()); n_c = int((m.treatment == 0).sum())

if MLFLOW_OK:
    try:
        import mlflow
        with mlflow.start_run(run_name=f"{study_id}_did"):
            mlflow.log_params({"study_id": study_id, "model_type": "did", "matching_method": pe_config["matching_method"]})
            mlflow.log_metrics({"estimate": round(point, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    except Exception as e:
        print("mlflow skipped:", e)

write_result("did", pe_config["matching_method"], round(point, 3), std_error=round(se, 3),
             ci_low=round(lo, 3), ci_high=round(hi, 3), n_treated=n_t, n_control=n_c)

log_step("3", "Modeling (did)", "PASS",
         pass_description=f"DiD estimate={point:,.2f} CI[{lo:,.1f},{hi:,.1f}]",
         n_treated=n_t, execution_time=time.time() - t0)

dbutils.notebook.exit("None")
