"""
AEE Executive Walkthrough — Databricks App entrypoint.

Serves a single self-contained, clickable walkthrough of the Automated
Evaluation Engine (static HTML). No data connection, no credentials, no PHI —
the story runs on illustrative synthetic figures, so it is safe to deploy and
share in any workspace.

The same static/index.html also opens directly in a browser (double-click) with
no server, which is handy for a quick screen-share.
"""
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AEE Executive Walkthrough")

# Serve everything in ./static at the root; html=True makes "/" return index.html.
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    # Databricks Apps injects the port to listen on via DATABRICKS_APP_PORT.
    port = int(os.getenv("DATABRICKS_APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
