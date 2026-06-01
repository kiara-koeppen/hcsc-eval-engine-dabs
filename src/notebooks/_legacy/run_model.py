# Databricks notebook source
# MAGIC %md
# MAGIC # [LEGACY] run_model
# MAGIC Shared by all four model tasks (`model_exact_att`, `model_exact_did`,
# MAGIC `model_knn_att`, `model_knn_did`) via different `base_parameters`. Even sharing the
# MAGIC file, each combination needs its **own task + its own routing condition** in the DAG —
# MAGIC that's the M x N blowup. Here: 2 matching x 2 models = 4 model tasks.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "STUDY_001", "Study ID")
dbutils.widgets.text("matching_label", "exact", "Matching label")
dbutils.widgets.text("model_method", "att", "Model method")
catalog, schema, study_id, matching_label, model_method = (
    dbutils.widgets.get(x) for x in ("catalog", "schema", "study_id", "matching_label", "model_method"))
print(f"[{study_id}] LEGACY model task: matching={matching_label} model={model_method}")

# COMMAND ----------

import numpy as np, pandas as pd
from datetime import datetime, timezone
from pyspark.sql.types import (StructType, StructField, StringType, LongType,
                               DoubleType, BooleanType)

m = spark.table(f"{catalog}.{schema}.matched_{study_id.lower()}").toPandas()
t, c = m[m.treatment == 1], m[m.treatment == 0]
wmean = lambda s, w: float(np.average(np.asarray(s, float), weights=np.asarray(w, float)))

if model_method == "att":
    est = wmean(t.post_outcome, t.weight) - wmean(c.post_outcome, c.weight)
elif model_method == "did":
    est = wmean(t.post_outcome - t.pre_outcome, t.weight) - wmean(c.post_outcome - c.pre_outcome, c.weight)
else:
    raise ValueError(model_method)

# Explicit schema — the legacy path leaves CI/SE columns null, and Spark Connect
# can't infer types for all-null columns, so we declare them.
result_schema = StructType([
    StructField("study_id", StringType()), StructField("run_ts", StringType()),
    StructField("matching_method", StringType()), StructField("model_method", StringType()),
    StructField("n_treated", LongType()), StructField("n_control", LongType()),
    StructField("estimate", DoubleType()), StructField("std_error", DoubleType()),
    StructField("ci_low", DoubleType()), StructField("ci_high", DoubleType()),
    StructField("naive_estimate", DoubleType()), StructField("significant", BooleanType()),
    StructField("source", StringType()),
])
row = (study_id, datetime.now(timezone.utc).isoformat(), matching_label, model_method,
       int(len(t)), int(len(c)), round(est, 3), None, None, None, None, None, "before")
spark.createDataFrame([row], schema=result_schema) \
     .write.mode("append").option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.evaluation_results")
print(f"[{study_id}] legacy estimate ({matching_label}/{model_method}) = {est:,.2f}")
