# Databricks notebook source
# MAGIC %md
# MAGIC # LEAF: feature_standard
# MAGIC
# MAGIC Standard-family feature engineering: single baseline + single performance period.
# MAGIC Reads `cohorts` for one study, imputes a covariate, scores data quality, and (only if
# MAGIC the study passes its quality threshold) writes `feat_<study_id>`.
# MAGIC
# MAGIC **This is a leaf notebook.** It is owned by whichever team maintains the standard
# MAGIC feature flow. It is invoked by `dispatch_feature` via `dbutils.notebook.run(...)`, and
# MAGIC returns the gate decision to the caller with `dbutils.notebook.exit("true"/"false")`.
# MAGIC The dispatcher promotes that string into the task value the quality gate reads.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_ATT_001", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")
print(f"[{study_id}] leaf=feature_standard")

# COMMAND ----------

import pandas as pd
import numpy as np

pdf = (spark.table(f"{catalog}.{schema}.cohorts")
            .where(f"study_id = '{study_id}'").toPandas())
cfg = (spark.table(f"{catalog}.{schema}.{config_table}")
            .where(f"study_id = '{study_id}'").collect()[0])
min_quality = float(cfg["min_quality_score"])

# COMMAND ----------

# Conservative global-median imputation. (Other feature leaves can impute differently.)
missing_fraction = float(pdf["risk_score"].isna().mean())
n_treated = int((pdf["treatment"] == 1).sum())
n_control = int((pdf["treatment"] == 0).sum())

features = pdf.copy()
features["risk_score"] = features["risk_score"].fillna(features["risk_score"].median())

# COMMAND ----------

# Quality score: completeness + sample adequacy, compared to the study's threshold.
completeness = 1.0 - missing_fraction
size_adequacy = min(1.0, min(n_treated, n_control) / 100.0)
quality_score = round(0.6 * completeness + 0.4 * size_adequacy, 3)
quality_passed = quality_score >= min_quality
print(f"[{study_id}] quality_score={quality_score} threshold={min_quality} -> "
      f"{'PASS' if quality_passed else 'FAIL'}")

if quality_passed:
    feat_table = f"{catalog}.{schema}.feat_{study_id.lower()}"
    (spark.createDataFrame(features)
          .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(feat_table))
    print(f"[{study_id}] wrote {feat_table}")
else:
    print(f"[{study_id}] below threshold — skipping downstream")

# COMMAND ----------

# Return the gate decision as the notebook's single output string. dispatch_feature
# captures this and sets the `quality_passed` task value the condition_task compares.
dbutils.notebook.exit(str(quality_passed).lower())
