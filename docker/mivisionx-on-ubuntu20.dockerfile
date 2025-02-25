FROM ubuntu:20.04

ARG ROCM_INSTALLER_REPO=https://repo.radeon.com/amdgpu-install/6.1.1/ubuntu/focal/amdgpu-install_6.1.60101-1_all.deb
ARG ROCM_INSTALLER_PACKAGE=amdgpu-install_6.1.60101-1_all.deb

ENV MIVISIONX_DEPS_ROOT=/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN apt-get update -y
# install mivisionx base dependencies - Level 1
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install gcc g++ cmake pkg-config git libcanberra-gtk-module
# install ROCm for mivisionx OpenCL/HIP dependency - Level 2
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration &&  \
        sudo apt-get -y clean && dpkg --add-architecture i386 && \
        wget ${ROCM_INSTALLER_REPO} && \
        sudo apt-get install -y ./${ROCM_INSTALLER_PACKAGE} && \
        sudo apt-get update -y && \
        sudo amdgpu-install -y --usecase=rocm
# install OpenCV & FFMPEG - Level 3
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/"
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
        libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip && \
        mkdir OpenCV && cd OpenCV && wget https://github.com/opencv/opencv/archive/refs/tags/4.6.0.zip && unzip 4.6.0.zip && \
        mkdir build && cd build && cmake -DWITH_GTK=ON -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_OPENCL=OFF ../opencv-4.6.0 && make -j8 && sudo make install && sudo ldconfig && cd
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install autoconf automake build-essential cmake git-core libass-dev libfreetype6-dev libsdl2-dev libtool libva-dev \
        libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo wget zlib1g-dev \
        nasm yasm libx264-dev libx265-dev libnuma-dev libfdk-aac-dev && \
        wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.4.2.zip && unzip n4.4.2.zip && cd FFmpeg-n4.4.2/ && sudo ldconfig && \
        ./configure --enable-shared --disable-static --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libass --enable-gpl --enable-nonfree && \
        make -j8 && sudo make install && cd
# install MIVisionX neural net dependency - Level 4
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install half rocblas-dev miopen-hip-dev migraphx-dev rocdecode-dev
# install MIVisionX AMD VX RPP dependency - Level 5
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install wget libbz2-dev libssl-dev python-dev python3-dev libgflags-dev libgoogle-glog-dev liblmdb-dev nasm yasm libjsoncpp-dev clang && \
        git clone -b 2.0.6.2 https://github.com/rrawther/libjpeg-turbo.git && cd libjpeg-turbo && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_STATIC=FALSE -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
        -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib ../ && make -j4 && sudo make install && cd ../../ && \
        git clone -b 1.5.0 https://github.com/ROCm/rpp.git && cd rpp && mkdir build && cd build && \
        cmake -DBACKEND=HIP ../ && make -j4 && sudo make install && cd ../../ && \
ENV CUPY_INSTALL_USE_HIP=1
ENV ROCM_HOME=/opt/rocm
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install git g++ hipblas hipsparse rocrand hipfft rocfft rocthrust-dev hipcub-dev python3 python3-pip python3-dev && \
        git clone -b v12.2.0 https://github.com/ROCmSoftwarePlatform/cupy.git && \
        git submodule update --init && pip install -e . --no-cache-dir -vvvv && pip install numpy==1.23.0 && cd

ENV MIVISIONX_WORKSPACE=/workspace
WORKDIR $MIVISIONX_WORKSPACE

ENV PATH=$PATH:/opt/rocm/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

# Clone MIVisionX 
RUN git clone https://github.com/ROCm/MIVisionX.git && \
        mkdir build && cd build && cmake -DBACKEND=HIP ../MIVisionX && make -j8 && make install