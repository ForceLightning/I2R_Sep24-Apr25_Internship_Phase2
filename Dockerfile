ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
ARG PYTHON_VERSION=3.12

# (1): Install base nvidia cuda container.
FROM ${BASE_IMAGE} AS dev-base
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        gcc && \
    rm -rf /var/lib/apt/lists/*
    
# (2): Download `uv` installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer and remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`.
ENV PATH="/root/.local/bin/:$PATH"

RUN uv python install 3.12
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache 
RUN --mount=type=cache,target=/opt/ccache/ ccache --set-config=cache_dir=/opt/ccache

# (3.1): Setup work directory for project.
ARG USER_TZ="Asia/Singapore"
ENV PYTHONPATH="src/:thirdparty/VPS"
ENV TZ=${USER_TZ}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /code
COPY ./pyproject.toml ./pyproject.toml
COPY ./uv.lock ./uv.lock

# (3.2): Install dependencies
RUN uv python pin 3.12
RUN uv venv --python 3.12
RUN uv sync --all-extras
RUN . ./.venv/bin/activate

# (4): Setup build process for OpenCV with CUDA support.
ARG OPENCV_VERSION=4.11.0.86
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -ex \
    && apt-get -qq update \
    && apt-get install -y --no-install-recommends \
        wget unzip \
        libhdf5-103-1 libhdf5-dev \
        libopenblas0 libopenblas-dev \
        libprotobuf32 libprotobuf-dev \
        libjpeg8 libjpeg8-dev \
        libpng16-16 libpng-dev \
        libtiff6 libtiff-dev \
        libwebp7 libwebp-dev \
        libopenjp2-7 libopenjp2-7-dev \
        libtbb12 libtbb-dev \
        libeigen3-dev \
        tesseract-ocr tesseract-ocr-por libtesseract-dev

# Clone Repository
RUN git clone --recursive https://github.com/opencv/opencv-python.git
WORKDIR /opt/opencv-python
RUN git checkout 86

# Build OpenCV
RUN set -ex \
    && ENABLE_HEADLESS=1 ENABLE_CONTRIB=1 \
        CMAKE_ARGS="-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=ON -DMKL_USE_MULTITHREAD=ON -DPYTHON3_NUMPY_INCLUDE_DIRS=/code/.venv/lib/python3.12/site-packages/numpy/_core/include" \
        MAKEFLAGS="-j$(nproc)" \
        uv build --wheel .

RUN apt-get -qq autoremove \
    && apt-get -qq clean

# (5): Install OpenCV
WORKDIR /code
RUN uv remove opencv-contrib-python-headless --group opencv-custom-build
# RUN uv add --group opencv-custom-build "opencv-contrib-python-headless @ opencv_contrib_python_headless-4.11.0.86-cp312-cp312-linux_x86_64.whl" -v
COPY /opt/opencv-python/dist/opencv_contrib_python_headless-4.11.0.86-cp312-cp312-linux_x86_64.whl* .
RUN uv add "opencv-contrib-python-headless @ /opt/opencv-python/dist/opencv_contrib_python_headless-4.11.0.86-cp312-cp312-linux_x86_64.whl" --group opencv-custom-build
ENV DOCKERZIED=1
