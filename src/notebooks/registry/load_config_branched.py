# Databricks notebook source
# MAGIC %md
# MAGIC # load_config_branched — choose what runs, then set the branch flags
# MAGIC
# MAGIC Reads `study_config_modular` and decides which studies run and which branches fire.
# MAGIC
# MAGIC Selection logic:
# MAGIC - If the `study_ids` parameter is non-empty (comma-separated), run **exactly those**
# MAGIC   studies (on-demand subset / single-study runs, including re-running an archived study).
# MAGIC - Otherwise run every study with `active = true`. Inactive rows stay in the table for
# MAGIC   reference but are skipped (the "don't run" flag).
# MAGIC
# MAGIC Then it splits the selected studies into the standard for_each list and per-LME-model
# MAGIC flags, so the job's condition tasks only fire the branches that have work to do.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")
dbutils.widgets.text("study_ids", "", "Optional comma-separated study_ids to run (overrides active)")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
config_table = dbutils.widgets.get("config_table")
study_ids_raw = dbutils.widgets.get("study_ids").strip()

# COMMAND ----------

from pyspark.sql import functions as F

cfg = spark.table(f"{catalog}.{schema}.{config_table}")

# `active` may not exist on older config tables; treat missing as active=true.
if "active" not in cfg.columns:
    cfg = cfg.withColumn("active", F.lit(True))

if study_ids_raw:
    wanted = [s.strip() for s in study_ids_raw.split(",") if s.strip()]
    selected = cfg.where(F.col("study_id").isin(wanted))
    print(f"study_ids override -> running exactly: {wanted}")
else:
    selected = cfg.where(F.col("active") == True)  # noqa: E712
    print("no study_ids override -> running all active studies")

rows = selected.collect()
families = [(r["model_family"] or "standard") for r in rows]

standard_studies = [
    {"study_id": r["study_id"], "feature_nb": r["feature_nb"], "matching_nb": r["matching_nb"],
     "matching_method": r["matching_method"], "model_nb": r["model_nb"]}
    for r in rows if (r["model_family"] or "standard") == "standard"
]

has_standard   = "true" if standard_studies else "false"
has_lme_mixed  = "true" if "lme_mixed"  in families else "false"
has_lme_growth = "true" if "lme_growth" in families else "false"

print(f"selected {len(rows)} studies: {[r['study_id'] for r in rows]}")
print(f"standard ({len(standard_studies)}) | has_lme_mixed={has_lme_mixed} | has_lme_growth={has_lme_growth}")

dbutils.jobs.taskValues.set(key="standard_studies", value=standard_studies)
dbutils.jobs.taskValues.set(key="has_standard", value=has_standard)
dbutils.jobs.taskValues.set(key="has_lme_mixed", value=has_lme_mixed)
dbutils.jobs.taskValues.set(key="has_lme_growth", value=has_lme_growth)
