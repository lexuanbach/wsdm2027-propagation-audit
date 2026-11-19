# CPU image for the WSDM 2027 propagation audit.
#   docker build -t propagation-audit .
#   docker run --rm -it propagation-audit make test verify figures
# Mount data to rerun experiments, e.g.
#   docker run --rm -it -v "$PWD/data:/workspace/data" -v "$PWD/external:/workspace/external" \
#       propagation-audit make check-data
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends git make unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements-lock.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple -r requirements-lock.txt
COPY . .
RUN pip install --no-deps -e .

CMD ["make", "help"]
