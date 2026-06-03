# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 4 — Insights (modular)
# MAGIC
# MAGIC Same as the demo insights stage, but reads study metadata from `study_config_modular`.
# MAGIC Writes a plain-language interpretation of this study's latest result to
# MAGIC `evaluation_insights`.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_ATT_001", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Row
from datetime import datetime, timezone

res = (spark.table(f"{catalog}.{schema}.evaluation_results")
            .where(F.col("study_id") == study_id)
            .orderBy(F.col("run_ts").desc()).limit(1).collect()[0])
cfg = (spark.table(f"{catalog}.{schema}.{config_table}")
            .where(F.col("study_id") == study_id).collect()[0])

est = res["estimate"]
direction = "reduction" if est < 0 else "increase"
sig = "statistically significant" if res["significant"] else "not statistically significant"

headline = (f"{cfg['study_name']} ({cfg['vendor']}): {res['model_method'].upper()} on "
            f"{res['matching_method']}-matched cohort estimates a {abs(est):,.0f} {direction} "
            f"in the outcome ({sig}).")
interpretation = (
    f"Estimate {est:,.1f} (95% CI [{res['ci_low']:,.1f}, {res['ci_high']:,.1f}]). "
    f"Naive unadjusted comparison was {res['naive_estimate']:,.1f}; matching/weighting "
    f"shifted the estimate by {est - res['naive_estimate']:,.1f}. "
    f"n={res['n_treated']} treated / {res['n_control']} control."
)
print(headline)
print(interpretation)

spark.createDataFrame([Row(
    study_id=study_id, run_ts=datetime.now(timezone.utc).isoformat(),
    headline=headline, interpretation=interpretation,
)]).write.mode("append").option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.evaluation_insights")
print(f"[{study_id}] insight written")
