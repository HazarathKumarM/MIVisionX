FROM ubuntu:18.04

ENV MIVISIONX_DEPS_ROOT=/opt/mivisionx-deps
WORKDIR $MIVISIONX_DEPS_ROOT

RUN apt-get update -y
# install mivisionx base dependencies - Level 1
RUN apt-get -y install gcc g++ cmake git
# install ROCm for mivisionx OpenCL/HIP dependency - Level 2
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install initramfs-tools libnuma-dev wget sudo keyboard-configuration &&  \
        wget https://repo.radeon.com/amdgpu-install/22.20/ubuntu/bionic/amdgpu-install_22.20.50200-1_all.deb && \
        sudo apt-get -y install ./amdgpu-install_22.20.50200-1_all.deb && \
        sudo apt-get update -y && \
        sudo amdgpu-install -y --usecase=graphics,rocm

WORKDIR /workspace

# install MIVisionX
RUN git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX.git && mkdir build && cd build && \
        cmake -DBACKEND=OCL ../MIVisionX && make -j8 && make install