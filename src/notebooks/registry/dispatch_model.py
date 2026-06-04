# Databricks notebook source
# MAGIC %md
# MAGIC # DISPATCHER: model stage
# MAGIC
# MAGIC Runs the config-named model leaf (`model_nb`) with `dbutils.notebook.run(...)`. Each
# MAGIC model is its own notebook owned by its own team (`model_att`, `model_did`, `model_lme_mixed`,
# MAGIC ...). Adding the next model is a new leaf + a config row — never an edit to this file or
# MAGIC the DAG.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_ATT_001", "Study ID")
dbutils.widgets.text("model_nb", "model_att", "Model leaf notebook")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
model_nb = dbutils.widgets.get("model_nb")
config_table = dbutils.widgets.get("config_table")
print(f"[{study_id}] dispatch_model -> running leaf '{model_nb}'")

# COMMAND ----------

estimate = dbutils.notebook.run(
    model_nb,
    3600,
    {
        "catalog": catalog,
        "schema": schema,
        "study_id": study_id,
        "config_table": config_table,
    },
)
print(f"[{study_id}] leaf '{model_nb}' returned estimate={estimate}")

dbutils.jobs.taskValues.set(key="estimate", value=str(estimate))
