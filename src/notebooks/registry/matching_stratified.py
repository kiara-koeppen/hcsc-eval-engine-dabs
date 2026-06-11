# Databricks notebook source
# MAGIC %md
# MAGIC # LEAF: matching_stratified  (a "third team" plug-in)
# MAGIC
# MAGIC This notebook demonstrates the modular promise from the meeting: a different team
# MAGIC contributes a **new matching methodology** by dropping in one notebook and registering
# MAGIC it (`matching_nb = matching_stratified` in `study_config_modular`). Nobody else's code
# MAGIC changes, and the orchestration DAG does not change.
# MAGIC
# MAGIC Method: stratify on age x risk buckets, then within each stratum weight treated/control
# MAGIC so every stratum contributes equally (a simple stratification estimator). Reads
# MAGIC `feat_<study_id>`, writes `matched_<study_id>` with a `weight` column — same contract as
# MAGIC `matching_standard`, so the downstream model leaves consume it unchanged.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_STRAT_003", "Study ID")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")

print(f"[{study_id}] leaf=matching_stratified")

# COMMAND ----------

import numpy as np
import pandas as pd

df = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}").toPandas()

# Build strata on age x risk buckets.
df = df.copy()
df["age_b"] = pd.cut(df["age"], bins=[0, 50, 60, 70, 200], labels=False)
df["risk_b"] = pd.cut(df["risk_score"], bins=[0, 1, 1.5, 2, 100], labels=False)
df = df.dropna(subset=["age_b", "risk_b"])
df["stratum"] = df["age_b"].astype(int).astype(str) + "_" + df["risk_b"].astype(int).astype(str)

# Keep only strata that contain BOTH arms (otherwise no within-stratum comparison).
both = (df.groupby("stratum")["treatment"].nunique() == 2)
keep_strata = set(both[both].index)
df = df[df["stratum"].isin(keep_strata)].copy()

# Within each stratum, weight each arm so strata are balanced and each stratum's
# treated and control halves carry equal total weight.
def stratum_weights(g):
    w = pd.Series(1.0, index=g.index)
    for arm in (0, 1):
        m = g["treatment"] == arm
        cnt = int(m.sum())
        if cnt > 0:
            w[m] = 1.0 / cnt          # each arm in the stratum sums to weight 1
    return w

df["weight"] = (df.groupby("stratum", group_keys=False).apply(stratum_weights)
                  if len(df) else pd.Series(dtype=float))
matched = df.drop(columns=["age_b", "risk_b", "stratum"])

n_t = int((matched.treatment == 1).sum())
n_c = int((matched.treatment == 0).sum())
print(f"[{study_id}] stratified cohort: {n_t} treated / {n_c} control across {len(keep_strata)} strata")

# COMMAND ----------

out_table = f"{catalog}.{schema}.matched_{study_id.lower()}"
(spark.createDataFrame(matched)
      .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(out_table))
print(f"[{study_id}] wrote {out_table}")

dbutils.notebook.exit(f"{n_t},{n_c}")
