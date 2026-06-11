# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3: model_wilcoxon  (non-parametric pre/post test)
# MAGIC
# MAGIC No matching needed (Phase 1 routed straight here). Runs a paired pre/post Wilcoxon
# MAGIC signed-rank test on the treated group. The signed-rank test assumes the paired
# MAGIC differences are **symmetric**; if that assumption is violated (high skew), this notebook
# MAGIC routes to the configured `fallback_model` (the Sign Test, which needs no symmetry) by
# MAGIC setting nextphase to that notebook name -- HCSC's "Wilcoxon -> Sign Test on failure".

# COMMAND ----------

# MAGIC %run ./common_nb

# COMMAND ----------

import time, numpy as np
from pyspark.sql import Row
from datetime import datetime, timezone
t0 = time.time()

df = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}").toPandas()
treated = df[df.treatment == 1]
diffs = (treated["post_outcome"] - treated["pre_outcome"]).to_numpy(float)
# Symmetry check (skewness); dependency-free.
sd = diffs.std()
skew = float(np.mean(((diffs - diffs.mean()) / (sd + 1e-9)) ** 3))
print(f"[{study_id}] wilcoxon symmetry check: skew={skew:.3f} (|skew|>1 triggers fallback)")

# COMMAND ----------

if abs(skew) > 1.0:
    fallback = pe_config.get("fallback_model") or "None"
    log_step("3", "Modeling (wilcoxon)", "ROUTED",
             pass_description=f"symmetry assumption violated (skew={skew:.2f}); routing to {fallback}",
             n_treated=int(len(diffs)), execution_time=time.time() - t0)
    print(f"[{study_id}] assumption failed -> nextphase={fallback}")
    dbutils.notebook.exit(fallback)

# Assumption OK: run the signed-rank test.
try:
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(diffs)
    p = float(p)
except Exception as e:
    print("scipy.wilcoxon unavailable, recording estimate only:", e); p = None
estimate = float(np.median(diffs))

write_result("wilcoxon", "none", round(estimate, 3),
             p_value=(round(p, 5) if p is not None else None), n_treated=int(len(diffs)))

log_step("3", "Modeling (wilcoxon)", "PASS",
         pass_description=f"median diff={estimate:,.1f} p={p}", n_treated=int(len(diffs)),
         execution_time=time.time() - t0)
dbutils.notebook.exit("None")
