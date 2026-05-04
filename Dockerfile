# BrainSpace Docker image.
#
# Build: docker build -t brainspace .
# Run a Jupyter server: docker run --rm -p 8888:8888 brainspace
# Run a shell: docker run --rm -it brainspace bash
#
# This is a minimal headless image. Plotting works through vtk-osmesa
# (software rendering, no GPU required). For GPU-accelerated rendering
# swap vtk-osmesa for vtk-egl and run with --gpus all.
#
# Modern replacement for the 2020-vintage Neurodocker image proposed in #70.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime libraries VTK needs even with software rendering.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libxrender1 \
       libxext6 \
       libsm6 \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN useradd --create-home --shell /bin/bash brainspace
USER brainspace
WORKDIR /home/brainspace
ENV PATH="/home/brainspace/.local/bin:${PATH}"

# Install BrainSpace + headless VTK + Jupyter from the in-repo source.
COPY --chown=brainspace:brainspace . /home/brainspace/BrainSpace
RUN pip install --user --upgrade pip \
    && pip install --user \
       jupyterlab \
       nilearn \
       /home/brainspace/BrainSpace[examples] \
    && pip uninstall -y vtk \
    && pip install --user vtk-osmesa

WORKDIR /home/brainspace/BrainSpace

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", \
     "--ServerApp.token=", "--ServerApp.password="]
