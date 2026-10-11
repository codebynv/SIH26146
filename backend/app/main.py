"""FastAPI entry point for BitTrace AI (SIH26146).

Steps 1-3 foundations plus the Step 9A dashboard API and the static hosting
needed to serve the Step 9A/9B investigator dashboard: a health surface, the
ingestion endpoint, and the unified domain model.  No correlation, graph, ML or
persistence lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.dashboard.api import router as dashboard_router
from backend.app.ingestion.parsers import IngestionError
from backend.app.ingestion.service import IngestionService
from backend.app.ingestion.uploads import (
    MAX_UPLOAD_BYTES,
    UploadError,
    discard_upload,
    store_upload,
)

app = FastAPI(
    title="BitTrace AI Backend",
    version="0.1.0",
    description="FastAPI entry point for the BitTrace AI investigation platform.",
)

# Include Step 9A dashboard router (read-only)
app.include_router(dashboard_router)


_service = IngestionService()

#: The vanilla HTML/CSS/JS dashboard, served from the repository itself.
import os

_main_path = Path(__file__).resolve()
FRONTEND_DIR = _main_path.parents[2] / "frontend"

if not (FRONTEND_DIR / "index.html").is_file():
    # Fallback to current working directory if not found in parents[2]
    alt_dir = Path(os.getcwd()) / "frontend"
    if (alt_dir / "index.html").is_file():
        FRONTEND_DIR = alt_dir

DASHBOARD_HTML = FRONTEND_DIR / "index.html"


@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Investigator dashboard",
    response_description="The Step 9A/9B investigator dashboard (static HTML).",
)
async def dashboard() -> Response:
    """Serve the investigator dashboard.

    The dashboard is same-origin static HTML/CSS/JS, so it has to be served from
    this application rather than opened from ``file://``: its ES modules and its
    ``/api/dashboard/*`` fetches only work over HTTP from one origin.

    Deployments that deliberately ship without the frontend -- the Step 10
    runtime image excludes it -- keep the original Step 1 greeting at the root
    instead of a broken page.
    """
    if not DASHBOARD_HTML.is_file():
        return JSONResponse({"message": "Welcome to BitTrace AI backend"})
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


@app.get("/api/welcome", summary="Service greeting")
async def welcome() -> dict[str, str]:
    """Return the Step 1 service greeting.

    The root path now serves the dashboard, so the plain JSON greeting lives at
    an explicit API path.
    """
    return {"message": "Welcome to BitTrace AI backend"}


@app.get("/health", summary="Health check endpoint")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "BitTrace AI"}


def _summarise(result: Any) -> dict[str, Any]:
    """Render an ingestion result without leaking internal objects."""
    return {
        "ingestion_id": result.ingestion_id,
        "source_format": result.source_format.value,
        "raw_record_count": result.raw_record_count,
        "record_count": result.record_count,
        "duplicate_count": result.duplicate_count,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "warnings": list(result.warnings),
        "issues": [issue.model_dump(mode="json") for issue in result.issues],
        "records": [
            {
                "record": record.model_dump(mode="json"),
                "is_valid": record.is_valid,
            }
            for record in result.records
        ],
    }


@app.post("/ingest", summary="Ingest a CSV, JSON or XML dataset")
async def ingest(
    file: UploadFile = File(..., description="CSV, JSON or XML dataset."),
    declared_format: Optional[str] = Form(
        default=None,
        description="Optional format hint. Cross-checked against the detected "
        "content; a mismatch is rejected.",
    ),
) -> dict[str, Any]:
    """Ingest an uploaded dataset and return the normalised records.

    The file is written to a server-generated temporary path, parsed by the
    format detected from its content, and the temporary file is removed before
    the response is returned.
    """
    content = await file.read()
    try:
        stored = store_upload(
            content,
            client_filename=file.filename,
            declared_format=declared_format,
        )
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = _service.ingest_path(
            stored.path,
            declared_format=stored.detected_format,
            context={"client_filename": stored.display_filename},
            # This endpoint created the staging file, so this endpoint owns it
            # and may remove it.  Ownership is explicit, never inferred.
            owned=True,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        # Idempotent belt-and-braces for a failure that happened before
        # ingest_path could take ownership of the staged file.
        discard_upload(stored.path)

    return _summarise(result)


# --------------------------------------------------------------------------- #
# Static dashboard assets
#
# Mounted last, after every API route, so no mount can shadow /api/dashboard,
# /health, /ingest or the OpenAPI documents. The dashboard is referenced with
# relative URLs, so /css and /js must resolve at the root of the origin; the
# whole tree is also mounted at /dashboard for the /dashboard/ entry point.
# A deployment without the repository checkout (the Step 10 image) simply does
# not serve them.
# --------------------------------------------------------------------------- #

if (FRONTEND_DIR / "css").is_dir():
    app.mount(
        "/css",
        StaticFiles(directory=FRONTEND_DIR / "css"),
        name="dashboard-css",
    )

if (FRONTEND_DIR / "js").is_dir():
    app.mount(
        "/js",
        StaticFiles(directory=FRONTEND_DIR / "js"),
        name="dashboard-js",
    )

if FRONTEND_DIR.is_dir():
    app.mount(
        "/dashboard",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="dashboard",
    )


__all__ = ["app", "MAX_UPLOAD_BYTES", "FRONTEND_DIR", "DASHBOARD_HTML"]

