# Databricks notebook source
# MAGIC %md
# MAGIC # DISPATCHER: matching stage
# MAGIC
# MAGIC Runs the config-named matching leaf (`matching_nb`) with `dbutils.notebook.run(...)`,
# MAGIC passing the chosen `matching_method`. A team contributing a new matching method just
# MAGIC adds a leaf notebook and registers its name in `study_config_modular`; this dispatcher
# MAGIC and the job DAG are untouched.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "M_ATT_001", "Study ID")
dbutils.widgets.text("matching_nb", "matching_standard", "Matching leaf notebook")
dbutils.widgets.text("matching_method", "exact", "Matching method (for the leaf)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
matching_nb = dbutils.widgets.get("matching_nb")
matching_method = dbutils.widgets.get("matching_method")
print(f"[{study_id}] dispatch_matching -> running leaf '{matching_nb}' (method={matching_method})")

# COMMAND ----------

counts = dbutils.notebook.run(
    matching_nb,
    3600,
    {
        "catalog": catalog,
        "schema": schema,
        "study_id": study_id,
        "matching_method": matching_method,
    },
)
print(f"[{study_id}] leaf '{matching_nb}' returned n_treated,n_control = {counts}")

# Surface counts as task values (handy for monitoring / downstream conditions).
try:
    n_t, n_c = [int(x) for x in str(counts).split(",")]
    dbutils.jobs.taskValues.set(key="n_treated", value=n_t)
    dbutils.jobs.taskValues.set(key="n_control", value=n_c)
except Exception as e:
    print(f"[{study_id}] could not parse counts '{counts}': {e}")
