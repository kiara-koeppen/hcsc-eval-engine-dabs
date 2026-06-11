# Databricks notebook source
# MAGIC %md
# MAGIC # DISPATCHER: matching stage  (config-table driven, no per-value widgets)
# MAGIC
# MAGIC The job passes only the KEY (`study_id`). This dispatcher reads BOTH the matching
# MAGIC notebook name (`matching_nb`) and the chosen `matching_method` from `study_config_modular`,
# MAGIC then runs the named leaf. Adding/retargeting a matching method is a config-table edit,
# MAGIC not a widget change or a DAG change.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_ATT_001", "Study ID")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")

# COMMAND ----------

cfg = (spark.table(f"{catalog}.{schema}.{config_table}")
            .where(f"study_id = '{study_id}'").collect()[0])
matching_nb = cfg["matching_nb"]
print(f"[{study_id}] dispatch_matching -> config says matching_nb='{matching_nb}'")

# COMMAND ----------

# The leaf reads its own matching_method from the config table; we pass only the key.
counts = dbutils.notebook.run(
    matching_nb,
    3600,
    {"catalog": catalog, "schema": schema, "study_id": study_id, "config_table": config_table},
)
print(f"[{study_id}] leaf '{matching_nb}' returned n_treated,n_control = {counts}")

try:
    n_t, n_c = [int(x) for x in str(counts).split(",")]
    dbutils.jobs.taskValues.set(key="n_treated", value=n_t)
    dbutils.jobs.taskValues.set(key="n_control", value=n_c)
except Exception as e:
    print(f"[{study_id}] could not parse counts '{counts}': {e}")
