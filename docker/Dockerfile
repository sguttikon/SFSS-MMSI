ARG CUDA_VERSION=11.7.1
ARG OS_VERSION=22.04
# pull a prebuilt image
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}

# setup timezone
ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "-c"]

ARG USER_ID
ARG GROUP_ID

# Required to build Ubuntu 22.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

# Create a non-root user and home directory
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --home "/home/user" --uid $USER_ID --gid $GROUP_ID user
RUN chown -R user:user /home/user

# Set the working directory to /home/user
ENV DIRPATH /home/user
WORKDIR $DIRPATH

# Switch to root user
USER root

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential \
    curl \
    ca-certificates \
    libx11-6 \
    nano \
    graphviz \
    libgl1-mesa-glx \
    openssh-server \
    apt-transport-https

# Install other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
    libsm6 libxext6 libxrender-dev \
    libgtk2.0-dev pkg-config \
    libopenmpi-dev \
 && sudo rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install Conda
ENV PATH /opt/miniconda/bin:${PATH}
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --no-check-certificate && /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    conda init bash && \
    conda config --set auto_activate_base false && \
    conda update --all && \
    rm ~/miniconda.sh && conda clean -ya

# Create myenv environment
ENV CONDA_ENVIRONMENT_PATH /opt/miniconda/envs/myenv
ENV CONDA_DEFAULT_ENV myenv
RUN conda create -y -n ${CONDA_DEFAULT_ENV} python=3.8
ENV PATH ${CONDA_ENVIRONMENT_PATH}/bin:${PATH}
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Enable myenv environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# ln -sf is needed to make sure that version `GLIBCXX_3.4.30' is found
RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_ENVIRONMENT_PATH}/bin/../lib/libstdc++.so.6

# install pytorch
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# install nvcc v11.7
RUN conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit

# install ninja to speed up the compilation
RUN pip install ninja

# install mmcv from source
RUN git clone https://github.com/open-mmlab/mmcv.git --branch v1.6.2 --single-branch && \
    cd mmcv && \
    MMCV_WITH_OPS=1 pip install -e .

# install other dependencies
RUN pip install timm==0.6.12 \
                mmsegmentation==0.30.0 \
                fvcore==0.1.5.post20221221 \
                open3d==0.16.0 \
                easydict==1.10 \
                opencv-python==4.7.0.68 \
                tensorboardx==2.5.1 \
                notebook==7.0.2

# install py360convert from source
RUN git clone -b sfss_mmsi https://github.com/sguttikon/py360convert.git && \
    pip install -e py360convert

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA="1"
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH
ENV CFLAGS="-I$CUDA_HOME/include $CFLAGS"
ENV PATH=/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/miniconda3/envs/bin:$PATH

RUN python -c 'import torch; print(f"PyTorch version: {torch.__version__}"); print(f"CUDA available:  {torch.cuda.is_available()}"); print(f"cuDNN version:   {torch.backends.cudnn.version()}"); print(torch.__config__.show());'

USER user
RUN ["/bin/bash"]