# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11

# *********************** Base *************************
# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm AS base

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

ENV PYTHONUNBUFFERED=1

# Install required system libs for OpenCV / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# *********************** Production *************************
FROM base AS production

# Copy project code into the image
COPY . /app

# Install project itself (non-editable, no dev deps)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Run the worker (via the script defined in pyproject.toml)
CMD ["worker"]

# *********************** Development *************************
FROM base AS development

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev

CMD ["uv", "run", "--group", "dev", "watchfiles", "--filter", "python", "worker"]
