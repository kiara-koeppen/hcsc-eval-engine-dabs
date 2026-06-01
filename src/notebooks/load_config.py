# Databricks notebook source
# MAGIC %md
# MAGIC # load_config — emit the study list for `for_each`
# MAGIC
# MAGIC Reads `study_config` and publishes the rows as a **task value** (`studies`).
# MAGIC The downstream `for_each` task iterates that array, launching one
# MAGIC `run_single_study` run per row.
# MAGIC
# MAGIC This is the entire "control plane" for fan-out: add a row to `study_config`
# MAGIC and the next run automatically picks it up. No DAG edits, ever.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

rows = spark.table(f"{catalog}.{schema}.study_config").collect()

# Each element becomes one for_each iteration. Field names here are referenced
# in the job YAML as {{input.study_id}}, {{input.matching_method}}, etc.
studies = [
    {
        "study_id": r["study_id"],
        "feature_method": r["feature_method"],
        "matching_method": r["matching_method"],
        "model_method": r["model_method"],
    }
    for r in rows
]

print(f"Publishing {len(studies)} studies for fan-out:")
for s in studies:
    print(" ", s)

# Publish as a task value -> consumed by the for_each task's `inputs`.
dbutils.jobs.taskValues.set(key="studies", value=studies)
