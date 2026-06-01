# HCSC Automated Evaluation Engine — DABs Demo

A Databricks Asset Bundle (DAB) demo built for HCSC's Program Evaluation / Health
Economics team. It shows how to orchestrate a config-driven causal **program-evaluation
engine** (data processing → matching → modeling → insights) in a way that **scales flat**
as you add matching methods, evaluation models, and studies — eliminating the
exponential If/else-task branching the team is hitting today.

> **The problem we're solving.** Today the team's job emits a "next phase" value from a
> notebook, and an **If/else condition task per candidate notebook** routes true/false to
> the matching path, then to the model path. Every new matching method or model adds a new
> task + a new condition + DAG rewiring, and model routing gets copy-pasted under each
> matching branch — an `M × N` blowup. With a target of ~10 matching methods × ~20 models,
> that's hundreds of hand-wired tasks **per study**.
>
> **The fix.** Drive selection with a **config table**, dispatch on a **parameter** inside
> a single notebook per stage, and fan out across studies with a **`for_each`** task. The
> DAG stays at a fixed 3 stages no matter how many methods, models, or studies you add. The
> only condition task that remains is the legitimate **pass/fail data-quality gate**.

---

## What's in the box

Two jobs you can run side by side to make the contrast obvious in a demo:

| Job | Pattern | Point |
|-----|---------|-------|
| **`eval_engine_before`** | An If/else condition task per candidate notebook (⚠️ current state) | Watch the DAG: 2 matching × 2 models already = 4 model tasks + 4 routing conditions, most **skipped** every run. Doesn't scale; no study-level fan-out. |
| **`eval_engine_after`** | `load_config` → `for_each(studies)` → `run_single_study` (✅ scalable) | Flat DAG. Add a study = one config row. Add a method/model = one branch in a notebook. `for_each` fans out (and parallelizes) over all studies. |
| `run_single_study` | The reusable per-study pipeline invoked by `for_each` | feature_engineering → **quality_gate** → matching → modeling → insights |
| `setup_eval_engine` | One-time data setup | Builds `study_config` + synthetic `cohorts` |

### The "after" DAG (flat, regardless of scale)

```
load_config ──▶ for_each(studies, concurrency=5) ──▶ run_single_study (per study)
                                                         │
   feature_engineering ─▶ quality_gate ─true─▶ matching ─▶ modeling ─▶ insights
                              └─false─▶ (study stops; nothing downstream runs)
```

`matching` and `modeling` are **single parameterized notebooks**. `matching_method`
(`exact`|`knn`|`ipw`) and `model_method` (`att`|`ate`|`did`|`gee`|`mixed`) select the
algorithm *inside* the notebook — not via a different task.

### The "before" DAG (grows with every notebook)

```
prep_and_route ─▶ gate ─true─┬─▶ route_exact ─true─▶ exact_matching ─┬─▶ route_exact_att ─true─▶ model (exact+att)
                             │                                       └─▶ route_exact_did ─true─▶ model (exact+did)
                             └─▶ route_knn   ─true─▶ knn_matching   ─┬─▶ route_knn_att   ─true─▶ model (knn+att)
                                                                     └─▶ route_knn_did   ─true─▶ model (knn+did)
```

The model layer is **duplicated under every matching branch** (a converging task can't read
a task value from a sibling branch that got skipped). That's the `M × N` explosion.

---

## Why DABs (the orchestration value)

- **Define once, deploy anywhere.** `databricks.yml` parameterizes `catalog`/`schema` per
  target (`dev` / `test` / `prod`). Promote the same code across environments by changing
  variables, not code.
- **Parameterized notebooks.** Notebooks read `dbutils.widgets` (catalog, schema, study_id,
  method choices). The job YAML feeds those widgets from job parameters / config — exactly
  the "the YAML feeds the pipeline" idea from our call.
- **Resource globbing.** `include: resources/*.yml` — drop a new `*.job.yml` in `resources/`
  and it's part of the bundle. No central registry to maintain.
- **Source-controlled + reproducible.** The whole pipeline (jobs + notebooks + config) is
  one versioned bundle you `deploy` and `run` with a single command.

---

## Architecture / data flow

```
study_config ──(load_config)──▶ studies[]  ──for_each──▶  per study:
cohorts ──▶ feature_engineering ─▶ feat_<study>  ─▶ matching ─▶ matched_<study> ─▶ modeling ─▶ evaluation_results ─▶ insights ─▶ evaluation_insights
                     │
                     └─ quality gate: writes quality_passed task value; gate short-circuits bad-data studies
```

**Tables** (in `${catalog}.${schema}`):
- `study_config` — one row per evaluation; the 3 method columns drive dispatch.
- `cohorts` — synthetic patient-level treatment/control data for all studies.
- `feat_<study_id>`, `matched_<study_id>` — per-study intermediate outputs.
- `evaluation_results` — one row per completed study (`source` = `before`/`after`).
- `evaluation_insights` — plain-language interpretation per study.

The synthetic data bakes in a known treatment effect per study and makes treated patients
sicker (confounding), so matching/weighting actually changes the estimate vs. a naive
comparison. **`STUDY_006`** has tiny n + heavy missingness and is intentionally **failed by
the quality gate** to demonstrate the short-circuit.

---

## Prerequisites

- Databricks CLI **v0.230+** (built with v0.292.0). `databricks --version`
- A profile in `~/.databrickscfg` (this demo uses **`kk_test`**).
- Serverless compute enabled in the workspace (jobs run serverless — no cluster config).
- The target catalog (`kk_test`) must already exist; the setup notebook creates the schema.

### ⚠️ Known CLI hiccup: Terraform download

Some CLI versions fail to download Terraform with
`unable to verify checksums signature: openpgp: key expired`. Point the CLI at a local
Terraform binary (`brew install terraform`):

```bash
export DATABRICKS_TF_EXEC_PATH=/opt/homebrew/bin/terraform
export DATABRICKS_TF_VERSION=1.12.1
```

---

## Deploy & run

```bash
# from repo root
export DATABRICKS_TF_EXEC_PATH=/opt/homebrew/bin/terraform DATABRICKS_TF_VERSION=1.12.1  # if needed

databricks bundle validate -t dev
databricks bundle deploy   -t dev

# 1) one-time: build config + synthetic data
databricks bundle run setup_eval_engine -t dev

# 2) the scalable pattern — fans out over all studies
databricks bundle run eval_engine_after -t dev

# 3) (optional) the current-state pattern, for contrast — runs one study
databricks bundle run eval_engine_before -t dev
```

Inspect results:

```sql
SELECT * FROM kk_test.eval_engine_demo.evaluation_results ORDER BY source, study_id;
SELECT * FROM kk_test.eval_engine_demo.evaluation_insights ORDER BY study_id;
```

Deploy to another environment by switching the target (`-t test` / `-t prod`); only the
`catalog`/`schema` variables change.

### Scale knobs to show in the demo
- **Add a study:** insert a row into `study_config`, re-run `eval_engine_after`. The DAG is
  unchanged; `for_each` just iterates one more time.
- **Add a matching method (e.g. `psm`):** add a branch in `src/notebooks/matching.py`. No
  job edits.
- **Add a model (e.g. `synthetic_control`):** add a branch in `src/notebooks/modeling.py`.
  No job edits.
- **Parallelism:** `for_each_task.concurrency` in `resources/eval_engine_after.job.yml`.

---

## File / folder structure

```
hcsc-eval-engine-dabs/
├── databricks.yml                       # bundle config + dev/test/prod targets + variables
├── resources/
│   ├── setup.job.yml                    # config + synthetic data job
│   ├── run_single_study.job.yml         # per-study pipeline (the for_each unit)
│   ├── eval_engine_after.job.yml        # ✅ config + for_each (scalable)
│   └── eval_engine_before.job.yml       # ⚠️ if/else explosion (current state)
└── src/
    ├── setup/
    │   └── 00_create_config_and_data.py # study_config + synthetic cohorts
    └── notebooks/
        ├── load_config.py               # emits studies[] task value for for_each
        ├── feature_engineering.py       # dispatcher + quality gate (emits quality_passed)
        ├── matching.py                  # dispatcher: exact | knn | ipw
        ├── modeling.py                  # dispatcher: att | ate | did | gee | mixed (+ bootstrap CI)
        ├── insights.py                  # plain-language interpretation
        └── _legacy/                     # notebooks used by the "before" job
            ├── prep_and_route.py        # emits chosen_matching / chosen_model / quality_passed
            ├── exact_matching.py
            ├── knn_matching.py
            └── run_model.py             # shared by all 4 model tasks (att/did)
```

---

## Configuration / variables

| Variable | Where | Default (dev) | Notes |
|----------|-------|---------------|-------|
| `catalog` | `databricks.yml` `variables` / per target | `kk_test` | UC catalog for all assets |
| `schema`  | `databricks.yml` `variables` / per target | `eval_engine_demo` (dev), `_test`, `_prod` | per-environment schema |
| job params | each `*.job.yml` | passed to notebook widgets | `study_id`, `feature_method`, `matching_method`, `model_method` |
| `for_each_task.concurrency` | `eval_engine_after.job.yml` | `5` | parallel studies |

Notebooks never hardcode catalog/schema — all read via `dbutils.widgets`.

---

## Notes / caveats

- The estimators here (ATT/ATE/DiD/GEE/mixed) are intentionally lightweight (numpy/pandas,
  dependency-free) so the demo runs anywhere on serverless. They illustrate the
  orchestration pattern, not production-grade causal inference — in production these stage
  notebooks would call the team's real models. DiD studies (STUDY_002, STUDY_005) best show
  the estimate diverging from the naive number as confounding is removed.
- Built and validated against the `kk_test` workspace, June 2026.
