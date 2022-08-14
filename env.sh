# Required packages
# apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev python3-pip
# LLVM
# pip3 install --user numpy decorator attrs scipy tornado psutil xgboost cloudpickle mxnet-cu102

export TVM_HOME=/home/shiyuw3/Research/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
export CUDA_HOME=/usr/local/cuda/
export PATH=/usr/local/NVIDIA-Nsight-Compute:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
