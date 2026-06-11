# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1: Feature Engineering & Data Standardization
# MAGIC
# MAGIC Entry point. Reads raw `pe_cohorts` for this study, standardizes into a cohort table,
# MAGIC runs a data-quality gate, then sets the **nextphase** signal:
# MAGIC - quality fails -> `None` (stop)
# MAGIC - model needs matching -> `matching`
# MAGIC - model needs no matching -> `model_<model_type>` (skip Phase 2)
# MAGIC
# MAGIC Depends on config only.

# COMMAND ----------

# MAGIC %run ./common_nb

# COMMAND ----------

import time, pandas as pd, numpy as np
t0 = time.time()

pdf = (spark.table(f"{catalog}.{schema}.pe_cohorts").where(f"study_id = '{study_id}'").toPandas())
n_input = len(pdf)
min_quality = float(pe_config["min_quality_score"])

missing = float(pdf["risk_score"].isna().mean())
pdf["risk_score"] = pdf["risk_score"].fillna(pdf["risk_score"].median())
n_t = int((pdf.treatment == 1).sum()); n_c = int((pdf.treatment == 0).sum())

quality = round(0.6 * (1 - missing) + 0.4 * min(1.0, min(n_t, n_c) / 100.0), 3)
passed = quality >= min_quality

if passed:
    spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true") \
        .saveAsTable(f"{catalog}.{schema}.feat_{study_id.lower()}")

# COMMAND ----------

# Decide the next phase from the config (conditional matching).
if not passed:
    nextphase = "None"
elif bool(pe_config["needs_matching"]):
    nextphase = "matching"
else:
    nextphase = f"model_{pe_config['model_type']}"   # skip Phase 2

log_step("1", "Feature Engineering & Data Standardization",
         "PASS" if passed else "FAIL",
         pass_description=f"quality={quality} threshold={min_quality}; nextphase={nextphase}",
         error_msg=None if passed else f"quality {quality} below {min_quality}",
         n_input=n_input, n_treated=n_t, n_columns=pdf.shape[1], execution_time=time.time() - t0)

dbutils.notebook.exit(nextphase)
