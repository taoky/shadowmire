ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.12.5

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM docker.io/library/python:${PYTHON_VERSION}-slim AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/shadowmire

WORKDIR /app

# Install locked runtime dependencies in a layer that remains cached when only
# the application source changes.
COPY pyproject.toml uv.lock README.md LICENSE LICENSE.AFL ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM docker.io/library/python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/shadowmire/bin:${PATH}" \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/shadowmire /opt/shadowmire

# Mount the mirror repository here, or override the working directory at run
# time. Shadowmire uses its current directory when --repo is not specified.
WORKDIR /mirror

ENTRYPOINT ["shadowmire"]
CMD ["--help"]
