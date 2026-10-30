# Multi-stage build. PyGambit 16.7.0 ships no Linux/macOS wheels: build it
# from sdist in the builder with matching Python headers and a C++ toolchain.
# GPL-2.0-or-later review gate: do NOT push this image (any registry) before
# the license review with required notices and source availability.
FROM python:3.12 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir Cython setuptools wheel

# Download sdist/wheels and build pygambit into a wheelhouse
WORKDIR /build
RUN pip download pygambit==16.7.0 --no-deps --no-binary :all: -d /build/sdist \
    && pip wheel --no-deps -w /wheelhouse /build/sdist/pygambit-16.7.0.tar.gz
RUN pip download open_spiel==2.0.2 scipy==1.18.1 numpy==2.5.4 \
        cryptography==50.0.2 sympy==1.14.0 -d /wheelhouse

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY --from=builder /wheelhouse /wheelhouse
RUN pip install --no-cache-dir --find-links /wheelhouse -r requirements.txt \
    && rm -rf /wheelhouse

COPY advisor.py game_models.py game_lp.py game_solvers.py game_estimation.py \
     game_simulation.py game_solver_worker.py solver_worker.py case_store.py \
     share_report.py plain_view.py social_media_gt.py webapp.py mcp_server.py ./
COPY LICENSE THIRD_PARTY_NOTICES.md ./

EXPOSE 7860
CMD ["python", "webapp.py", "--host", "0.0.0.0", "--port", "7860"]
