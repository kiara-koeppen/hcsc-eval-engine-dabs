# Databricks notebook source
# MAGIC %md
# MAGIC # DISPATCHER: feature stage  (config-table driven, no per-value widgets)
# MAGIC
# MAGIC The job passes only the KEY (`study_id`) plus where the config lives. This dispatcher
# MAGIC then READS the per-study settings from `study_config_modular` (the `feature_nb` to run),
# MAGIC rather than receiving them as widgets. This mirrors HCSC's pattern: values live in a
# MAGIC config table and the notebooks pull them, instead of being typed into widgets.
# MAGIC
# MAGIC It runs the config-named leaf with `dbutils.notebook.run(...)` (job `notebook_path` is
# MAGIC static, so the notebook is selected in code), and promotes the leaf's gate decision into
# MAGIC the `quality_passed` task value the downstream condition task compares.

# COMMAND ----------

# Only the KEY + location are widgets. Everything study-specific comes from the table.
dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_ATT_001", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")

# COMMAND ----------

# Read this study's row from the config table and pull the feature notebook name.
cfg = (spark.table(f"{catalog}.{schema}.{config_table}")
            .where(f"study_id = '{study_id}'").collect()[0])
feature_nb = cfg["feature_nb"]
print(f"[{study_id}] dispatch_feature -> config says feature_nb='{feature_nb}'")

# COMMAND ----------

# Run the config-named leaf. Leaves live as siblings in this registry/ folder.
quality_passed = dbutils.notebook.run(
    feature_nb,
    3600,
    {"catalog": catalog, "schema": schema, "study_id": study_id, "config_table": config_table},
)
print(f"[{study_id}] leaf '{feature_nb}' returned quality_passed={quality_passed}")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="quality_passed", value=str(quality_passed).strip().lower())
