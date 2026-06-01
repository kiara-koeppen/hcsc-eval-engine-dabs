# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 4 — Insights
# MAGIC
# MAGIC Reads this study's latest result and writes a plain-language interpretation to
# MAGIC `evaluation_insights`. In production this is where the "insights layer" / front end
# MAGIC would source its narrative.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "STUDY_001", "Study ID")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")

# COMMAND ----------

from pyspark.sql import functions as F

res = (spark.table(f"{catalog}.{schema}.evaluation_results")
            .where(F.col("study_id") == study_id)
            .orderBy(F.col("run_ts").desc())
            .limit(1).collect()[0])
cfg = spark.table(f"{catalog}.{schema}.study_config").where(F.col("study_id") == study_id).collect()[0]

est = res["estimate"]
direction = "reduction" if est < 0 else "increase"
sig = "statistically significant" if res["significant"] else "not statistically significant"

headline = (f"{cfg['study_name']} ({cfg['vendor']}): {res['model_method'].upper()} on "
            f"{res['matching_method']}-matched cohort estimates a {abs(est):,.0f} {direction} "
            f"in the outcome ({sig}).")
interpretation = (
    f"Estimate {est:,.1f} (95% CI [{res['ci_low']:,.1f}, {res['ci_high']:,.1f}]). "
    f"Naive unadjusted comparison was {res['naive_estimate']:,.1f}; matching/weighting "
    f"shifted the estimate by {est - res['naive_estimate']:,.1f}, removing confounding bias. "
    f"n={res['n_treated']} treated / {res['n_control']} control."
)
print(headline)
print(interpretation)

# COMMAND ----------

from pyspark.sql import Row
from datetime import datetime, timezone
spark.createDataFrame([Row(
    study_id=study_id, run_ts=datetime.now(timezone.utc).isoformat(),
    headline=headline, interpretation=interpretation,
)]).write.mode("append").option("mergeSchema", "true").saveAsTable(f"{catalog}.{schema}.evaluation_insights")
print(f"[{study_id}] insight written")
