# Databricks notebook source
# MAGIC %md
# MAGIC # DISPATCHER: model stage  (config-table driven, no per-value widgets)
# MAGIC
# MAGIC The job passes only the KEY (`study_id`) plus `job_run_id` for the reproducibility stamp.
# MAGIC This dispatcher reads the model notebook name (`model_nb`) from `study_config_modular`
# MAGIC and runs it. Each model is its own team-owned notebook; adding one is a config-table row,
# MAGIC never an edit to this file or the DAG.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_ATT_001", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")
dbutils.widgets.text("job_run_id", "", "Job run id (reproducibility stamp)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")
job_run_id = dbutils.widgets.get("job_run_id")

# COMMAND ----------

cfg = (spark.table(f"{catalog}.{schema}.{config_table}")
            .where(f"study_id = '{study_id}'").collect()[0])
model_nb = cfg["model_nb"]
print(f"[{study_id}] dispatch_model -> config says model_nb='{model_nb}'")

# COMMAND ----------

estimate = dbutils.notebook.run(
    model_nb,
    3600,
    {"catalog": catalog, "schema": schema, "study_id": study_id,
     "config_table": config_table, "job_run_id": job_run_id},
)
print(f"[{study_id}] leaf '{model_nb}' returned estimate={estimate}")

dbutils.jobs.taskValues.set(key="estimate", value=str(estimate))
