# syntax=docker/dockerfile:1
# PocketSlam CPU image. On macOS run natively instead: containers cannot use
# the Apple GPU (MPS). See RULES.md section 6.

FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

FROM python:3.12-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS runtime
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data/outputs /app/data/recordings \
    && chown -R app:app /app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app web/ /app/web/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    POCKETSLAM_HOST=0.0.0.0 \
    POCKETSLAM_PORT=10001 \
    POCKETSLAM_WEB_DIR=/app/web \
    POCKETSLAM_OUTPUT_DIR=/app/data/outputs \
    POCKETSLAM_RECORDINGS_DIR=/app/data/recordings
USER app
EXPOSE 10001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "pocketslam.healthcheck"]
CMD ["python", "-m", "pocketslam"]
