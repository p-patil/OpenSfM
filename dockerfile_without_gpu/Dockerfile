FROM ubuntu:16.04

# Install OpenCV from source, including contrib modules
## Install apt-getable dependencies
RUN apt-get update && apt-get install -y \
        awscli \
        build-essential \
        cmake \
        ffmpeg \
        git \
        pkg-config \
        python-dev \
        python-pip \
        unzip \
        vim \
        tmux \
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
    numba \
    sklearn

# Set up globus connect
RUN yes | pip install globus-cli
RUN useradd --create-home --home-dir /home/piyush --shell /bin/bash piyush
USER piyush
RUN cd /home/piyush && \
    wget -O /home/piyush/globusconnectpersonal-latest.tgz https://s3.amazonaws.com/connect.globusonline.org/linux/stable/globusconnectpersonal-latest.tgz && \
    tar -xzf globusconnectpersonal-latest.tgz && \
    globus login --no-local-server && \
    SETUP_KEY=$(globus endpoint create --personal Thinkpad-T440-Piyush | grep "Setup Key:" | cut -d ":" -f 2 | tr -d '[:space:]') && \
    globusconnectpersonal-2.3.3/globusconnectpersonal -setup $SETUP_KEY
