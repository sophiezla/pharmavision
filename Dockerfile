# Regenerates every committed table and figure from the committed run folders (PLAN 10.3):
#   docker build -t pharmachecked .
#   docker run --rm -v "$PWD/results:/repo/results" pharmachecked
# then `git diff --stat results/tables` should be empty. Images, weights and the locked test
# set are not needed and not included.
FROM python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY pyproject.toml README.md ./
COPY constraints ./constraints
COPY src ./src
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        -c constraints/cpu.txt \
    && pip install -e . -c constraints/cpu.txt
COPY . .
CMD ["python", "scripts/reproduce.py"]
