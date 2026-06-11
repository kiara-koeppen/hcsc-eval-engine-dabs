# Databricks notebook source
# MAGIC %md
# MAGIC # run_pe_study — orchestrator (nextphase router)
# MAGIC
# MAGIC Realizes HCSC's routing model: "condition tasks evaluate nextphase and determine flow...
# MAGIC model routing handled by setting nextphase to the next notebook name." Each phase notebook
# MAGIC returns the name of the next notebook to run (or `None`), and this router runs it. The
# MAGIC same `orchestration_run_id` is threaded into every step, so all steps share one run id and
# MAGIC log to `pe_step_log` under it.
# MAGIC
# MAGIC The chain per study, driven entirely by the config + nextphase signals:
# MAGIC   feature_engineering -> (matching | model_<type>) -> [matching -> model_<type>] ->
# MAGIC   model_<type> -> (None | fallback model e.g. model_sign_test) -> None
# MAGIC
# MAGIC The flow is NOT hard-wired here: which model runs, whether matching runs, and whether a
# MAGIC fallback fires are all decided by the config and the notebooks at runtime.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "PE_ATT_001", "Study ID (pe_jobname analog)")
dbutils.widgets.text("config_table", "pe_study_config", "Config table")
dbutils.widgets.text("pe_environment", "dev", "Environment")
dbutils.widgets.text("run_type", "Pipeline", "Run type")
dbutils.widgets.text("orchestration_run_id", "", "Orchestration run id (set by the job)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")
pe_environment = dbutils.widgets.get("pe_environment")

# Same run id for every step in this study (HCSC hard requirement). The job passes
# {{job.run_id}}; for a manual run with no value, generate one so steps still share it.
orchestration_run_id = dbutils.widgets.get("orchestration_run_id")
if not orchestration_run_id:
    import uuid
    orchestration_run_id = "manual-" + uuid.uuid4().hex[:12]
print(f"[{study_id}] orchestration_run_id={orchestration_run_id}")

args = {"catalog": catalog, "schema": schema, "study_id": study_id, "config_table": config_table,
        "pe_environment": pe_environment, "orchestration_run_id": orchestration_run_id}

# COMMAND ----------

nextnb = "feature_engineering"
trace = []
for _ in range(12):  # safety bound
    if not nextnb or nextnb.strip().lower() in ("none", ""):
        break
    step = nextnb.strip()
    trace.append(step)
    print(f"[{study_id}] --> running phase notebook: {step}")
    nextnb = dbutils.notebook.run(step, 3600, args)
    nextnb = (nextnb or "None").strip()

print(f"[{study_id}] pipeline complete. phase chain: {' -> '.join(trace)} -> done")
dbutils.notebook.exit(" -> ".join(trace))
