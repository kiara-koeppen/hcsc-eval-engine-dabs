# Databricks notebook source
# MAGIC %md
# MAGIC # load_config_branched — split studies by family, then BRANCH
# MAGIC
# MAGIC This is the control point for the branched design the team asked for. It reads
# MAGIC `study_config_modular` and splits the rows by `model_family`:
# MAGIC
# MAGIC - **standard** studies (ATT, ATE, DID, pre/post — single baseline + performance period)
# MAGIC   go down the shared `for_each` loop.
# MAGIC - **lme** (and any other structurally different family) studies go down their **own
# MAGIC   separate workstream** with dedicated notebooks.
# MAGIC
# MAGIC It publishes two lists plus two boolean flags. The job uses the flags in condition
# MAGIC tasks to branch from config into the right workstream(s).

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
config_table = dbutils.widgets.get("config_table")

# COMMAND ----------

rows = spark.table(f"{catalog}.{schema}.{config_table}").collect()

# Standard family -> shared for_each flow. Carry the leaf names the dispatchers need.
standard_studies = [
    {
        "study_id": r["study_id"],
        "feature_nb": r["feature_nb"],
        "matching_nb": r["matching_nb"],
        "matching_method": r["matching_method"],
        "model_nb": r["model_nb"],
    }
    for r in rows if (r["model_family"] or "standard") != "lme"
]

# LME family -> its OWN separate workstream (dedicated notebooks, no dispatcher).
lme_studies = [
    {"study_id": r["study_id"]}
    for r in rows if (r["model_family"] or "standard") == "lme"
]

has_standard = "true" if standard_studies else "false"
has_lme = "true" if lme_studies else "false"

print(f"standard ({len(standard_studies)}):")
for s in standard_studies:
    print("  ", s)
print(f"lme ({len(lme_studies)}):")
for s in lme_studies:
    print("  ", s)

# Two lists feed two separate for_each tasks; the flags drive the branch conditions.
dbutils.jobs.taskValues.set(key="standard_studies", value=standard_studies)
dbutils.jobs.taskValues.set(key="lme_studies", value=lme_studies)
dbutils.jobs.taskValues.set(key="has_standard", value=has_standard)
dbutils.jobs.taskValues.set(key="has_lme", value=has_lme)
