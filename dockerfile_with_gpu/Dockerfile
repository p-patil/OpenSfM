# This Dockerfile is intended to be built using the nvidia-docker wrapper tool

FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

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
    numba \
    sklearn


# Initialize OpenSfM
RUN mkdir -p /root/deepdrive && \
    cd /root/deepdrive && \
    git clone https://github.com/p-patil/OpenSfM.git && \
    git clone https://github.com/fyu/dilation.git && \
    cd OpenSfM && \
    python setup.py build && \
    ./update_paths "/root/deepdrive/OpenSfM" "/root/deepdrive/dilation"


# Personal stuff

RUN apt-get update && \
    apt-get install -y \
        curl \
        python3 \
        zsh

# Set up config files
RUN cd ~/ && \
    ## Pull config files
    git clone https://github.com/p-patil/dotfiles.git && \
    ## Set up zsh
    cp dotfiles/.zshrc ~/.zshrc && \
    ## Set up tmux
    git clone  https://github.com/gpakosz/.tmux.git && \
    cp dotfiles/.tmux.conf.local ~/.tmux.conf.local && \
    ln -s ~/.tmux/.tmux.conf ~/.tmux.conf && \
    ## Set up vim
    cp dotfiles/.vimrc ~/.vimrc && \
    ## Clean up
    rm -rf dotfiles

# Configure vim
RUN curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim && \
    mkdir -p ~/.vim/colors && \
    curl -o ~/.vim/colors/molokai.vim https://raw.githubusercontent.com/tomasr/molokai/master/colors/molokai.vim && \
    mkdir -p ~/.vim/tmp && \
    vim +PlugInstall +qall

# Install tmux 2.6 from source
RUN apt-get update && \
    ## Install
    apt-get install -y libevent-dev libncurses-dev xclip && \
    cd ~/ && \
    curl -O https://raw.githubusercontent.com/gurpreetatwal/dotfiles/master/install/tmux-256color.terminfo && \
    curl --location --silent --show-error https://github.com/tmux/tmux/releases/download/2.6/tmux-2.6.tar.gz | tar -xz -C /tmp && \
    cd /tmp/tmux-2.6 && ./configure && make && \
    cd /tmp/tmux-2.6 && make install && \
    tic -o ~/.terminfo ~/tmux-256color.terminfo && \
    ## Configure
    printf "\nexport TERM=\"xterm-256color\"\n" >> ~/.zshrc

# Configure zsh and oh-my-zsh
RUN env TERM="xterm-256color" zsh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" || true && \
    chsh -s $(which zsh) && \
    (mv ~/.zshrc.pre-oh-my-zsh ~/.zshrc || true) && \
    printf "\nexec zsh" >> ~/.bashrc && \
    touch ~/.z
