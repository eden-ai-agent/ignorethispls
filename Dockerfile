# Revolutionary O(1) Biometric Matching System
# Patent Pending - Michael Derrick Jagneaux
# Production Docker Configuration

FROM python:3.11-slim-bullseye

# Metadata
LABEL maintainer="Michael Derrick Jagneaux <contact@revolutionarybiometrics.com>"
LABEL description="World's First O(1) Biometric Matching System - Patent Pending"
LABEL version="1.0.0"
LABEL patent.status="pending"
LABEL performance.complexity="O(1)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV APP_HOME=/app
ENV USER_NAME=biometric
ENV USER_UID=1000

# Install system dependencies for OpenCV and PostgreSQL
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    # PostgreSQL client
    postgresql-client \
    libpq-dev \
    # Build tools
    gcc \
    g++ \
    cmake \
    pkg-config \
    # System utilities
    curl \
    wget \
    vim-tiny \
    htop \
    # Performance monitoring
    sysstat \
    procps \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r ${USER_NAME} --gid=${USER_UID} && \
    useradd -r -g ${USER_NAME} --uid=${USER_UID} --home-dir=${APP_HOME} --shell=/bin/bash ${USER_NAME}

# Create application directory
RUN mkdir -p ${APP_HOME} && chown -R ${USER_NAME}:${USER_NAME} ${APP_HOME}

# Set working directory
WORKDIR ${APP_HOME}

# Copy requirements first for better caching
COPY requirements.txt ${APP_HOME}/
COPY setup.py ${APP_HOME}/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn[gevent] && \
    # Clean pip cache
    pip cache purge

# Copy application code
COPY src/ ${APP_HOME}/src/
COPY config/ ${APP_HOME}/config/
COPY tests/ ${APP_HOME}/tests/
COPY static/ ${APP_HOME}/static/
COPY templates/ ${APP_HOME}/templates/

# Copy configuration files
COPY *.py ${APP_HOME}/
COPY *.md ${APP_HOME}/
COPY *.txt ${APP_HOME}/
COPY .env.example ${APP_HOME}/

# Create necessary directories
RUN mkdir -p ${APP_HOME}/logs \
             ${APP_HOME}/uploads \
             ${APP_HOME}/temp \
             ${APP_HOME}/data \
             ${APP_HOME}/backups \
             ${APP_HOME}/certificates

# Set proper permissions
RUN chown -R ${USER_NAME}:${USER_NAME} ${APP_HOME} && \
    chmod +x ${APP_HOME}/src/web/app.py

# Switch to non-root user
USER ${USER_NAME}

# Expose application port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment configuration
ENV FLASK_APP=src.web.app:create_app
ENV FLASK_ENV=production
ENV WORKERS=4
ENV TIMEOUT=120
ENV MAX_REQUESTS=1000
ENV MAX_REQUESTS_JITTER=100

# Performance optimizations
ENV OMP_NUM_THREADS=4
ENV OPENCV_LOG_LEVEL=ERROR
ENV PYTHONOPTIMIZE=1

# Default command with production WSGI server
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "4", \
     "--worker-class", "gevent", \
     "--worker-connections", "1000", \
     "--timeout", "120", \
     "--keepalive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--preload-app", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "src.web.app:create_app()"]

# Alternative development command
# CMD ["python", "-m", "src.web.app"]

# Performance monitoring command
# CMD ["python", "-m", "src.utils.performance_monitor"]

# Labels for container orchestration
LABEL com.revolutionarybiometrics.service="o1-biometric-api"
LABEL com.revolutionarybiometrics.environment="production"
LABEL com.revolutionarybiometrics.patent="pending"
LABEL com.revolutionarybiometrics.performance="o1-constant-time"