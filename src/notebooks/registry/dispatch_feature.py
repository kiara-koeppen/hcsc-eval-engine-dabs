# Databricks notebook source
# MAGIC %md
# MAGIC # DISPATCHER: feature stage
# MAGIC
# MAGIC This is the thin glue that makes the registry pattern work. It does NOT contain feature
# MAGIC logic. It reads the leaf notebook name from a job parameter (`feature_nb`, supplied per
# MAGIC study by the config table) and runs that leaf with `dbutils.notebook.run(...)`.
# MAGIC
# MAGIC Why a dispatcher at all? Databricks job `notebook_path` is static — it cannot be set from
# MAGIC a parameter. So to pick a notebook per study at runtime we run it from code, where the
# MAGIC path IS a runtime string. The leaf runs as its own child run (full visibility in the run
# MAGIC tree) and returns its gate decision via `dbutils.notebook.exit(...)`. This dispatcher
# MAGIC promotes that string into the `quality_passed` task value the quality gate compares.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_ATT_001", "Study ID")
dbutils.widgets.text("feature_nb", "feature_standard", "Feature leaf notebook")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
feature_nb = dbutils.widgets.get("feature_nb")
config_table = dbutils.widgets.get("config_table")
print(f"[{study_id}] dispatch_feature -> running leaf '{feature_nb}'")

# COMMAND ----------

# Run the config-named leaf. The path is relative to THIS notebook's folder, so leaves
# live as siblings in the same registry/ directory. Returns "true"/"false" (the gate).
quality_passed = dbutils.notebook.run(
    feature_nb,
    3600,
    {
        "catalog": catalog,
        "schema": schema,
        "study_id": study_id,
        "config_table": config_table,
    },
)
print(f"[{study_id}] leaf '{feature_nb}' returned quality_passed={quality_passed}")

# COMMAND ----------

# Promote the leaf's decision to a task value the downstream condition_task reads.
dbutils.jobs.taskValues.set(key="quality_passed", value=str(quality_passed).strip().lower())
