# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3 (fallback): model_sign_test
# MAGIC
# MAGIC The fallback target when Wilcoxon's symmetry assumption fails. A pure sign test on the
# MAGIC paired pre/post differences (uses only the sign of each difference, so no symmetry
# MAGIC assumption). Appends one record and exits `None`. This notebook is reached only by
# MAGIC nextphase routing from `model_wilcoxon`.

# COMMAND ----------

# MAGIC %run ./common_nb

# COMMAND ----------

import time, math, numpy as np
from pyspark.sql import Row
from datetime import datetime, timezone
t0 = time.time()

df = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}").toPandas()
treated = df[df.treatment == 1]
diffs = (treated["post_outcome"] - treated["pre_outcome"]).to_numpy(float)
n_pos = int((diffs > 0).sum()); n_neg = int((diffs < 0).sum()); n = n_pos + n_neg

# Two-sided sign test p-value via normal approximation (dependency-free).
if n > 0:
    z = (n_pos - n / 2) / math.sqrt(n / 4)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
else:
    p = None
estimate = float(np.median(diffs))
print(f"[{study_id}] sign test: {n_pos} pos / {n_neg} neg, median diff={estimate:.1f}, p={p}")

write_result("sign_test", "none", round(estimate, 3),
             p_value=(round(p, 5) if p is not None else None), n_treated=int(len(diffs)))

log_step("3b", "Modeling (sign_test, fallback)", "PASS",
         pass_description=f"median diff={estimate:,.1f} p={p} (fallback from wilcoxon)",
         n_treated=int(len(diffs)), execution_time=time.time() - t0)
dbutils.notebook.exit("None")
