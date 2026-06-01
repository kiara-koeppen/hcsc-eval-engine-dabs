# Databricks notebook source
# MAGIC %md
# MAGIC # [LEGACY] exact_matching
# MAGIC A standalone matching notebook — one of N. In the *before* job this is its own task,
# MAGIC reached only when `route_exact` evaluates true. Add a method => add another notebook
# MAGIC AND another condition task AND rewire the DAG.

# COMMAND ----------

dbutils.widgets.text("catalog", "kk_test", "Catalog")
dbutils.widgets.text("schema", "eval_engine_demo", "Schema")
dbutils.widgets.text("study_id", "STUDY_001", "Study ID")
catalog, schema, study_id = (dbutils.widgets.get(x) for x in ("catalog", "schema", "study_id"))

# COMMAND ----------

import pandas as pd
df = spark.table(f"{catalog}.{schema}.feat_{study_id.lower()}").toPandas()
t, c = df[df.treatment == 1].copy(), df[df.treatment == 0].copy()
for d in (t, c):
    d["age_b"] = pd.cut(d["age"], bins=[0, 50, 60, 70, 200], labels=False)
    d["risk_b"] = pd.cut(d["risk_score"], bins=[0, 1, 1.5, 2, 100], labels=False)
strata = set(map(tuple, t[["age_b", "risk_b"]].dropna().values)) & \
         set(map(tuple, c[["age_b", "risk_b"]].dropna().values))
keep = lambda d: d[[tuple(x) in strata for x in d[["age_b", "risk_b"]].values]]
out = pd.concat([keep(t), keep(c)], ignore_index=True).drop(columns=["age_b", "risk_b"])
out["weight"] = 1.0
(spark.createDataFrame(out).write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{catalog}.{schema}.matched_{study_id.lower()}"))
print(f"[{study_id}] exact matching -> {int((out.treatment==1).sum())}/{int((out.treatment==0).sum())}")
