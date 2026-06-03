# Databricks notebook source
# MAGIC %md
# MAGIC # load_config_modular — emit the study list for `for_each`
# MAGIC
# MAGIC Reads `study_config_modular` and publishes the rows as a task value (`studies`). Each
# MAGIC element carries the per-study leaf notebook names so the `for_each` task can pass them
# MAGIC into `run_study_modular`. Adding a study = one row here; the DAG never changes.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
config_table = dbutils.widgets.get("config_table")

# COMMAND ----------

rows = spark.table(f"{catalog}.{schema}.{config_table}").collect()

# Field names here are referenced in the job YAML as {{input.study_id}},
# {{input.feature_nb}}, {{input.matching_nb}}, {{input.matching_method}}, {{input.model_nb}}.
studies = [
    {
        "study_id": r["study_id"],
        "feature_nb": r["feature_nb"],
        "matching_nb": r["matching_nb"],
        "matching_method": r["matching_method"],
        "model_nb": r["model_nb"],
    }
    for r in rows
]

print(f"Publishing {len(studies)} studies for fan-out:")
for s in studies:
    print(" ", s)

dbutils.jobs.taskValues.set(key="studies", value=studies)
