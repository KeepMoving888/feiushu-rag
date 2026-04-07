FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=pypi.org
ARG REQUIREMENTS_FILE=requirements.api.txt

WORKDIR /app

COPY ${REQUIREMENTS_FILE} ./requirements.txt
RUN set -eux; \
    python -m pip install --upgrade pip setuptools wheel; \
    for i in 1 2 3; do \
      python -m pip install \
        --index-url ${PIP_INDEX_URL} \
        --trusted-host ${PIP_TRUSTED_HOST} \
        --retries 20 \
        --timeout 180 \
        --no-cache-dir \
        -r requirements.txt && break; \
      echo "pip install failed (attempt ${i}), retrying in 8s..."; \
      sleep 8; \
      if [ "$i" = "3" ]; then \
        echo "mirror install failed, fallback to https://pypi.org/simple"; \
        python -m pip install \
          --index-url https://pypi.org/simple \
          --trusted-host pypi.org \
          --retries 20 \
          --timeout 180 \
          --no-cache-dir \
          -r requirements.txt; \
      fi; \
    done

COPY . .

EXPOSE 8511

CMD ["streamlit", "run", "admin.py", "--server.port=8511", "--server.address=0.0.0.0", "--server.headless=true"]
