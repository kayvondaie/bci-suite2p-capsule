# hash:sha256:0000000000000000000000000000000000000000000000000000000000000000
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.4.4-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    h5py==3.11.0 \
    matplotlib==3.9.4 \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    pandas==2.2.3 \
    scanimage-tiff-reader==1.4.1.4 \
    scipy==1.13.1 \
    suite2p==0.14.4 \
    tifffile==2024.8.30
