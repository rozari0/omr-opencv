FROM python:3.13-slim

# Install build dependencies and system libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libffi-dev \
    libssl-dev \
    ninja-build \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && export PATH="$HOME/.cargo/bin:$PATH" \
    && rm -rf /root/.cargo /root/.rustup

# Install uv via the official install script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Set environment variables
ENV PATH="/root/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT="/usr/local/"

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml /app/
RUN uv sync --no-dev

# Copy application files
COPY . /app/

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
