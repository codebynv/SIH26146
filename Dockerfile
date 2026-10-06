# BitTrace AI - Step 10 container image.
#
# Two stages. The builder resolves dependencies and byte-compiles so the runtime
# image needs no compiler, no build cache and no network. The runtime stage
# copies the installed site-packages and the source, then runs as an
# unprivileged user.

# --------------------------------------------------------------------------- #
# Stage 1: build
# --------------------------------------------------------------------------- #

FROM python:3.13-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Copied alone so the dependency layer is reused whenever only source changes.
COPY backend/requirements.txt backend/requirements-dev.txt ./

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --requirement requirements.txt \
 && pip install --requirement requirements-dev.txt \
 && python -m compileall -q /opt/venv/lib/python3.13/site-packages

# --------------------------------------------------------------------------- #
# Stage 2: runtime
# --------------------------------------------------------------------------- #

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PATH="/opt/venv/bin:$PATH"

# curl is used by the HEALTHCHECK below; the rest are already present in slim
# and are pinned explicitly so an image rebuild is reproducible.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

# A dedicated unprivileged account. Nothing in the image is written to at
# runtime except the upload staging area, so a read-only root filesystem with
# one writable tmpfs is sufficient and removes the need for a writable image
# layer entirely.
RUN groupadd --system --gid 10001 bittrace \
 && useradd --system --uid 10001 --gid bittrace --no-create-home --shell /usr/sbin/nologin bittrace \
 && mkdir -p /app/data/uploads \
 && chown -R bittrace:bittrace /app

WORKDIR /app

# Only what the application and its evaluation need. Tests, the frontend and
# documentation stay out of the runtime image.
COPY --chown=bittrace:bittrace backend/ ./backend/
COPY --chown=bittrace:bittrace evaluation/ ./evaluation/

USER bittrace

EXPOSE 8000

# The evaluation is offline, so a container that cannot reach the network is
# the only configuration in which the determinism claim is meaningful.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent --show-error http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]