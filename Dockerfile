# This Dockerfile is intended to be built using the nvidia-docker wrapper tool

FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# Install OpenCV from source, including contrib modules
## Install apt-getable dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        pkg-config \
        python-dev \
        python-pip \
        unzip \
        vim \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

## Install OpenCV dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get install -y libxvidcore-dev libx264-dev
RUN apt-get install -y libgtk-3-dev
RUN apt-get install -y libatlas-base-dev gfortran
RUN pip install --upgrade pip && pip install numpy

## Download OpenCV
RUN wget -O ~/opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip
RUN unzip -d ~/ ~/opencv.zip
RUN wget -O ~/opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
RUN unzip -d ~/ ~/opencv_contrib.zip

## Install OpenCV
RUN mkdir ~/opencv-3.1.0/build && \
    cd ~/opencv-3.1.0/build && \
    cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.1.0/modules \
        -D PYTHON_EXECUTABLE=/usr/bin/python \
        -D BUILD_EXAMPLES=ON \
        -D WITH_CUDA=OFF \
        .. && \
    make -j4 && \
    make install
RUN ldconfig

## Clean up
RUN rm ~/opencv.zip ~/opencv_contrib.zip

# Install Caffe
## Install apt-getable dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-numpy \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

## Set up Caffe build environment
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

ENV CLONE_TAG=1.0

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

## Done
WORKDIR /root

# Set up OpenSfM
## Install apt-getable dependencies
RUN apt-get update && \
    apt-get install -y \
        libatlas-base-dev \
        libboost-python-dev \
        libeigen3-dev \
        libgoogle-glog-dev \
        libsuitesparse-dev \
        python-pyexiv2 \
        python-pyproj \
        python-scipy \
        python-tk \
        python-yaml \
        unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

## Install Ceres from source
RUN \
    mkdir -p /source && cd /source && \
    wget http://ceres-solver.org/ceres-solver-1.10.0.tar.gz && \
    tar xvzf ceres-solver-1.10.0.tar.gz && \
    cd /source/ceres-solver-1.10.0 && \
    mkdir -p build && cd /source/ceres-solver-1.10.0/build && \
    cmake \
        -D CMAKE_C_FLAGS=-fPIC \
        -D CMAKE_CXX_FLAGS=-fPIC \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_TESTING=OFF \
        .. && \
    make install && \
    cd / && \
    rm -rf /source/ceres-solver-1.10.0 && \
    rm -f /source/ceres-solver-1.10.0.tar.gz

## Install OpenGV from source
### Download, make, and install. Note: in CMakeLists.txt, set the C++ standard to 11 for Python wrappers to work.
RUN \
    mkdir -p /source && cd /source && \
    git clone https://github.com/paulinus/opengv.git && \
    cd /source/opengv && \
    sed -i.bak "4i set (CMAKE_CXX_STANDARD 11)" CMakeLists.txt && \
    mkdir -p /source/opengv/build && cd /source/opengv/build && \
    cmake \
      -D BUILD_TESTS=OFF \
      -D BUILD_PYTHON=ON \
      .. && \
    make install && \
    cd / && \
    rm -rf /source/opengv

## Install necessary Python libraries
RUN pip install \
    matplotlib \
    networkx \
    exifread \
    xmltodict \
    pathlib2 \
    sklearn \
    numba
