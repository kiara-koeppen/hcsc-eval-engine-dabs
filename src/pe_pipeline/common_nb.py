# Databricks notebook source
# MAGIC %md
# MAGIC # common_nb — shared initialization (run via %run at the top of every phase notebook)
# MAGIC
# MAGIC Mirrors HCSC's `Config_Utility/common_nb`. Every phase/model notebook starts with
# MAGIC `%run ./common_nb`, which:
# MAGIC 1. (would) install required packages — here they're already on the cluster/serverless image.
# MAGIC 2. Reads the key params from widgets (set by the orchestrating job): `study_id`
# MAGIC    (HCSC's `pe_jobname`), `pe_environment`, `orchestration_run_id`.
# MAGIC 3. Loads this study's row from the config table into a `pe_config` dict.
# MAGIC 4. (would) retrieve SSO secrets from the configured scope — placeholder here.
# MAGIC 5. Defines `log_step(...)` which writes one row per step to the `pe_step_log` table,
# MAGIC    and sets up an MLflow experiment (best-effort).
# MAGIC
# MAGIC After `%run ./common_nb`, the caller has: `catalog`, `schema`, `study_id`,
# MAGIC `pe_config` (dict), `log_step(...)`, and `mlflow` configured.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "PE_ATT_001", "Study ID (HCSC pe_jobname analog)")
dbutils.widgets.text("config_table", "pe_study_config", "Config table")
dbutils.widgets.text("pe_environment", "dev", "Environment")
dbutils.widgets.text("orchestration_run_id", "", "Orchestration run id (same across all steps)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
study_id = dbutils.widgets.get("study_id")
config_table = dbutils.widgets.get("config_table")
pe_environment = dbutils.widgets.get("pe_environment")
orchestration_run_id = dbutils.widgets.get("orchestration_run_id")

# COMMAND ----------

# 3. Load this study's config row into pe_config (values come from the TABLE, not widgets).
_cfg_row = (spark.table(f"{catalog}.{schema}.{config_table}")
                 .where(f"study_id = '{study_id}'").collect()[0])
pe_config = _cfg_row.asDict()
print(f"[{study_id}] pe_config loaded from {config_table}: "
      f"model_type={pe_config.get('model_type')} needs_matching={pe_config.get('needs_matching')} "
      f"fallback_model={pe_config.get('fallback_model')}")

# 4. Secrets placeholder — HCSC's common_nb pulls SSO creds here:
#    sso_user = dbutils.secrets.get(scope=pe_config['secret_scope'], key='sso_user')
#    (omitted in this reference; no external system to authenticate to).

# COMMAND ----------

# 5a. log_step() + write_result(): one row per step / per result, with EXPLICIT schemas.
# (Spark Connect cannot infer types for one-row frames that contain None columns -> use
# explicit StructType. See feedback_spark_connect_all_null_columns.)
from datetime import datetime, timezone
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

_STEP_LOG = f"{catalog}.{schema}.pe_step_log"
_STEP_SCHEMA = StructType([
    StructField("run_id", StringType()), StructField("study_id", StringType()),
    StructField("step_id", StringType()), StructField("step_name", StringType()),
    StructField("status", StringType()), StructField("pass_description", StringType()),
    StructField("error_msg", StringType()), StructField("n_input", IntegerType()),
    StructField("n_treated", IntegerType()), StructField("n_columns", IntegerType()),
    StructField("execution_time", DoubleType()), StructField("pe_environment", StringType()),
    StructField("ts", StringType()),
])

def log_step(step_id, step_name, status, pass_description=None, error_msg=None,
             n_input=None, n_treated=None, n_columns=None, execution_time=None):
    """Append one row to pe_step_log, stamped with the shared orchestration_run_id."""
    data = [(orchestration_run_id, study_id, str(step_id), step_name, status,
             pass_description, error_msg,
             int(n_input) if n_input is not None else None,
             int(n_treated) if n_treated is not None else None,
             int(n_columns) if n_columns is not None else None,
             float(execution_time) if execution_time is not None else None,
             pe_environment, datetime.now(timezone.utc).isoformat())]
    (spark.createDataFrame(data, schema=_STEP_SCHEMA)
          .write.mode("append").option("mergeSchema", "true").saveAsTable(_STEP_LOG))
    print(f"[{study_id}] log_step {step_id} '{step_name}' -> {status}")

_RESULT_TBL = f"{catalog}.{schema}.pe_evaluation_results"
_RESULT_SCHEMA = StructType([
    StructField("run_id", StringType()), StructField("study_id", StringType()),
    StructField("model_method", StringType()), StructField("matching_method", StringType()),
    StructField("estimate", DoubleType()), StructField("std_error", DoubleType()),
    StructField("ci_low", DoubleType()), StructField("ci_high", DoubleType()),
    StructField("p_value", DoubleType()), StructField("n_treated", IntegerType()),
    StructField("n_control", IntegerType()), StructField("pe_environment", StringType()),
    StructField("ts", StringType()),
])

def write_result(model_method, matching_method, estimate, std_error=None, ci_low=None,
                 ci_high=None, p_value=None, n_treated=None, n_control=None):
    """Append one result row to pe_evaluation_results (explicit schema, None-safe)."""
    f = lambda v: (float(v) if v is not None else None)
    i = lambda v: (int(v) if v is not None else None)
    data = [(orchestration_run_id, study_id, model_method, matching_method, f(estimate),
             f(std_error), f(ci_low), f(ci_high), f(p_value), i(n_treated), i(n_control),
             pe_environment, datetime.now(timezone.utc).isoformat())]
    (spark.createDataFrame(data, schema=_RESULT_SCHEMA)
          .write.mode("append").option("mergeSchema", "true").saveAsTable(_RESULT_TBL))
    print(f"[{study_id}] wrote result: {model_method} estimate={estimate}")

# COMMAND ----------

# 5b. MLflow experiment (OPTIONAL observability, not core to the run).
# HCSC's common_nb %pip-installs mlflow/shap/flaml/linearmodels here; on this serverless
# image mlflow may be absent, so we treat it as optional. log_step (the table above) is the
# primary observability and always works.
MLFLOW_OK = False
try:
    import mlflow
    _exp = pe_config.get("experiment_name") or f"/Users/{spark.sql('select current_user()').collect()[0][0]}/pe_eval_experiment"
    mlflow.set_experiment(_exp)
    MLFLOW_OK = True
    print(f"[{study_id}] MLflow experiment: {_exp}")
except Exception as _e:
    print(f"[{study_id}] MLflow unavailable, skipping (log_step is primary observability): {_e}")
