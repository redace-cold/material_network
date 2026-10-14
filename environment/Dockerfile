FROM python:3.12.3-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/results/.matplotlib
COPY requirements.lock.txt /tmp/material-network-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/material-network-requirements.txt
WORKDIR /code
CMD ["bash", "run"]
