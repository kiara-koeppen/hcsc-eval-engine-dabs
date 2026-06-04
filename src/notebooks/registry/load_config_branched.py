# Databricks notebook source
# MAGIC %md
# MAGIC # load_config_branched — split by family, set the branch flags
# MAGIC
# MAGIC Reads `study_config_modular` and decides which branches the job should take:
# MAGIC
# MAGIC - **standard** studies go down the shared `for_each` loop. Their rows (with leaf names)
# MAGIC   are published as `standard_studies`.
# MAGIC - each **lme_*** family is its OWN branch with its own dedicated pipeline. We publish a
# MAGIC   boolean flag per LME model (`has_lme_mixed`, `has_lme_growth`) so the job's condition
# MAGIC   tasks fire only the branches that are actually configured.
# MAGIC
# MAGIC Adding another LME model = add its rows + a `has_<family>` flag here + a branch in the job.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("config_table", "study_config_modular", "Config table")
catalog = dbutils.widgets.get("catalog"); schema = dbutils.widgets.get("schema")
config_table = dbutils.widgets.get("config_table")

# COMMAND ----------

rows = spark.table(f"{catalog}.{schema}.{config_table}").collect()
families = [(r["model_family"] or "standard") for r in rows]

standard_studies = [
    {"study_id": r["study_id"], "feature_nb": r["feature_nb"], "matching_nb": r["matching_nb"],
     "matching_method": r["matching_method"], "model_nb": r["model_nb"]}
    for r in rows if (r["model_family"] or "standard") == "standard"
]

has_standard   = "true" if standard_studies else "false"
has_lme_mixed  = "true" if "lme_mixed"  in families else "false"
has_lme_growth = "true" if "lme_growth" in families else "false"

print(f"standard ({len(standard_studies)}):")
for s in standard_studies: print("  ", s)
print(f"has_standard={has_standard} has_lme_mixed={has_lme_mixed} has_lme_growth={has_lme_growth}")

dbutils.jobs.taskValues.set(key="standard_studies", value=standard_studies)
dbutils.jobs.taskValues.set(key="has_standard", value=has_standard)
dbutils.jobs.taskValues.set(key="has_lme_mixed", value=has_lme_mixed)
dbutils.jobs.taskValues.set(key="has_lme_growth", value=has_lme_growth)
