
# Tensorflow2-GPU-Training

```sh
# sudo apt install virtualenv

# set virtual environment
virtualenv --system-site-packages -p python3 ~/tensorflow-learn/tensorflow2-gpu-training/venv/tf2-gpu
source ~/tensorflow-learn/tensorflow2-gpu-training/venv/tf2-gpu/bin/activate
pip install tensorflow-gpu
python
import tensorflow as tf
tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)

# install driver from
# https://www.nvidia.com/Download/driverResults.aspx/151568/en-us

# install cuda 10 or 10.1
# download runfile local from
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal

# https://developer.nvidia.com/rdp/cudnn-archive
# Download cuDNN v7.6.3 (August 23, 2019), for CUDA 10.1
# or 
# Download cuDNN v7.6.3 (August 23, 2019), for CUDA 10.0

```

##### Dependencies
```sh

source ~/tensorflow-learn/tensorflow2-gpu-training/venv/tf2-gpu/bin/activate

pip install pydot
sudo apt-get install graphviz

pip install cifar2png
cd ~/datasets
# download cifar-10 dataaset
cifar2png cifar10 cifar-10

pip install scipy
pip install matplotlib
```
