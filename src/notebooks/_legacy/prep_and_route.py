# Databricks notebook source
# MAGIC %md
# MAGIC # [LEGACY] prep_and_route
# MAGIC
# MAGIC Current-state pattern: do feature engineering, then emit the "next phase" values
# MAGIC (`chosen_matching`, `chosen_model`) plus the quality gate flag. Downstream
# MAGIC **condition tasks** compare these strings to decide which branch runs.
# MAGIC
# MAGIC This is the notebook whose output the team described as driving the true/false routing.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "STUDY_001", "Study ID")
catalog, schema, study_id = (dbutils.widgets.get(x) for x in ("catalog", "schema", "study_id"))

# COMMAND ----------

import pandas as pd
pdf = spark.table(f"{catalog}.{schema}.cohorts").where(f"study_id='{study_id}'").toPandas()
cfg = spark.table(f"{catalog}.{schema}.study_config").where(f"study_id='{study_id}'").collect()[0]

# Feature engineering (impute) + quality gate
missing_fraction = float(pdf["risk_score"].isna().mean())
pdf["risk_score"] = pdf["risk_score"].fillna(pdf["risk_score"].median())
size_adequacy = min(1.0, min((pdf.treatment == 1).sum(), (pdf.treatment == 0).sum()) / 100.0)
quality_score = round(0.6 * (1 - missing_fraction) + 0.4 * size_adequacy, 3)
quality_passed = quality_score >= float(cfg["min_quality_score"])

if quality_passed:
    (spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true")
          .saveAsTable(f"{catalog}.{schema}.feat_{study_id.lower()}"))

# Emit the routing decisions — these are the "next phase" task values.
dbutils.jobs.taskValues.set(key="quality_passed", value=str(quality_passed).lower())
dbutils.jobs.taskValues.set(key="chosen_matching", value=cfg["matching_method"])
dbutils.jobs.taskValues.set(key="chosen_model", value=cfg["model_method"])
print(f"[{study_id}] quality_passed={quality_passed} "
      f"chosen_matching={cfg['matching_method']} chosen_model={cfg['model_method']}")
print("Downstream condition tasks will now branch on these strings.")
