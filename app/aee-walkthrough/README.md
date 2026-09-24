# AEE Executive Walkthrough (Databricks App)

An executive, click-through walkthrough of the **Automated Evaluation Engine (AEE)**: how a program evaluation that used to take 10 to 16 weeks of manual analysis now runs end to end in about 4 hours, as one automated, governed pipeline. It is built for a non-technical VP audience (Product, Corporate Strategy, AIS), leads with the front-end result, then shows how it works and why you can trust it, and closes on the strategy.

The app is a single self-contained HTML page served by a tiny FastAPI wrapper.

> **Safe to deploy and share anywhere.** The app has **no data connection, no credentials, and no PHI**. Every figure is illustrative synthetic data, and any number still awaiting a confirmed value is shown as a literal `X` placeholder. Nothing sensitive is embedded.

---

## What's in this folder

```
app/aee-walkthrough/
├── app.py            # FastAPI entrypoint; serves static/ at "/" (honors DATABRICKS_APP_PORT)
├── app.yaml          # Databricks Apps run command: ["python", "app.py"]
├── requirements.txt  # fastapi, uvicorn[standard]
├── static/
│   └── index.html    # the entire walkthrough (self-contained: HTML + CSS + JS, no build step)
└── README.md         # this file
```

There is no build step. Editing `static/index.html` is the whole content workflow.

---

## Quickest way to preview (no workspace needed)

Because the walkthrough is one self-contained file, you can just open it locally:

- **Double-click** `static/index.html` in a browser, or
- Run the server locally: `pip install -r requirements.txt && python app.py`, then open http://localhost:8000

---

## Prerequisites to deploy as a Databricks App

- A Databricks workspace with **Databricks Apps enabled**.
- Permission to create an App in that workspace.
- For the **CLI path**: the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/) installed and authenticated to your workspace (`databricks auth login --host <your-workspace-url>`), or a configured profile (`--profile <name>`).

Throughout, replace `you@example.com` with your workspace username and `<your-workspace-url>` with your host.

---

## Option A: Deploy from the CLI

Three steps: create the app, sync the code into the workspace, deploy.

```bash
# 1. Create the app compute (one time). Skip if the app already exists.
databricks apps create aee-walkthrough \
  --description "AEE executive walkthrough (synthetic / illustrative)"

# 2. Sync this folder up to a workspace path (one-way, local -> workspace).
#    Run this from inside app/aee-walkthrough/.
databricks sync . /Workspace/Users/you@example.com/apps/aee-walkthrough

# 3. Deploy from that workspace path.
databricks apps deploy aee-walkthrough \
  --source-code-path /Workspace/Users/you@example.com/apps/aee-walkthrough \
  --mode SNAPSHOT
```

When the deploy finishes, the CLI prints the app URL and status. Open the URL (you'll be prompted to sign in with your workspace identity).

**Redeploying after an edit:** re-run steps 2 and 3. To push local edits automatically during iterative work, use `databricks sync --watch .` together with `--mode AUTO_SYNC` on the deploy.

- `--mode SNAPSHOT` takes a one-time copy of the code at deploy time (use this for a stable demo).
- `--mode AUTO_SYNC` keeps the running app linked to the workspace files.

---

## Option B: Deploy from the UI

1. **Upload the code to your workspace.** In the workspace, go to **Workspace** in the left nav, open (or create) a folder such as `/Users/you@example.com/apps/aee-walkthrough`, click the kebab menu, choose **Import**, and upload the contents of this `app/aee-walkthrough/` folder (`app.py`, `app.yaml`, `requirements.txt`, and the `static/` folder). Keep the folder structure intact.
2. **Create the app.** In the left nav go to **Compute → Apps** (or **New → App**), click **Create app**, and choose the **Custom** option (deploy your own code).
3. **Point it at your code.** Give it a name (e.g. `aee-walkthrough`), and for the source select the workspace folder you imported in step 1.
4. **Deploy.** Click **Deploy**. When the status shows the app is running, open its URL.

To update later, re-import the changed files (step 1) and click **Deploy** again on the app.

---

## How it runs

`app.yaml` tells Databricks Apps to run `python app.py`. `app.py` starts FastAPI and mounts everything in `static/` at the root, so `/` returns `index.html`. Databricks Apps injects the port via the `DATABRICKS_APP_PORT` environment variable, which `app.py` reads.

## Editing the content

All copy, styling, and interactivity live in `static/index.html`. The page has four tabs (**The result**, **How it works**, **Why you can trust it**, **The strategy**) plus a clickable phase spine inside "How it works." Keep the VP-safe rules: only use numbers that have been confirmed for the VP audience, leave unconfirmed figures as `X`, and write in plain business language.
