FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/ubuntu:22.04-linuxarm64

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY config.json /app/config.json
# COPY models /app/models
COPY magic_pdf /app/
COPY 3rdparty/detectron2-0.6-cp310-cp310-linux_x86_64.whl /app/detectron2-0.6-cp310-cp310-linux_x86_64.whl

RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
COPY docker/sources.list /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-distutils \
        python3-pip \
        wget \
        git \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

COPY 3rdparty/eva_decord-0.6.1-cp310-cp310-macosx_10_9_x86_64.whl /app/eva_decord-0.6.1-cp310-cp310-macosx_10_9_x86_64.whl

RUN pip3 install /app/eva_decord-0.6.1-cp310-cp310-macosx_10_9_x86_64.whl

RUN pip3 install --upgrade pip && \
    pip3 install -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install /app/detectron2-0.6-cp310-cp310-linux_x86_64.whl && \
    pip3 cache purge

RUN rm -rf /app/requirements.txt && \
    rm -rf /app/detectron2-0.6-cp310-cp310-linux_x86_64.whl && \
    rm -rf /app/eva_decord-0.6.1-cp310-cp310-macosx_10_9_x86_64.whl

ENTRYPOINT ["/bin/bash", "-c", "python3 /app/app.py"]

