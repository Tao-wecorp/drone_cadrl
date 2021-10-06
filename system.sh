
# Install systemback
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 382003C2C8B7B4AB813E915B14E4942973C62A1B
sudo add-apt-repository "deb http://ppa.launchpad.net/nemh/systemback/ubuntu xenial main"
sudo apt update -y
sudo apt install systemback -y

# Install nvidia drivers
sudo dpkg -P $(dpkg -l | grep nvidia-driver | awk '{print $2}')
sudo apt autoremove
sudo reboot
sudo ubuntu-drivers autoinstall

# Install cuda 10.2 
# https://medium.com/@sh.tsang/tutorial-cuda-v10-2-cudnn-v7-6-5-installation-ubuntu-18-04-3d24c157473f
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

echo 'export PATH=/usr/local/cuda-10.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64' >> ~/.bashrc
source ~/.bashrc
nvcc -V

# # !!! Download cudnn first https://developer.nvidia.com/rdp/cudnn-archive
tar -zxf cudnn-10.2-linux-x64-v7.6.5.32.tgz
cd cuda
sudo cp -P lib64/* /usr/local/cuda/lib64/
sudo cp -P include/* /usr/local/cuda/include/
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
cd ~

# Install ros
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-desktop-full
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/melodic/lib" >> ~/.bashrc
source ~/.bashrc
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-joint-state-publisher-gui ros-melodic-teleop-twist-keyboard ffmpeg freeglut3-dev xvfb -y
pip3 install empy
sudo rosdep init
rosdep update
# nano ~/.ignition/fuel/config.yaml

# Install virualenv
cd ~/Installers/
wget https://bootstrap.pypa.io/get-pip.py
sudo apt-get install python3-distutils python3-apt
sudo python3 get-pip.py
sudo pip3 install virtualenv virtualenvwrapper
cd ~

echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc
echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' >> ~/.bashrc
echo 'source /usr/local/bin/virtualenvwrapper.sh' >> ~/.bashrc
source ~/.bashrc

mkvirtualenv py3venv -p python3 --system-site-packages
workon py3venv
pip3 install numpy defusedxml
pip3 install rospkg

# Install joystick
sudo apt-get install ros-melodic-joy
sudo apt install jstest-gtk
sudo pip install ds4drv
sudo ds4drv

# Install OpenCV
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install build-essential cmake unzip pkg-config libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev -y

cd ~/Installers/
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.2.0 opencv
mv opencv_contrib-4.2.0 opencv_contrib
cd ~

cd ~/Installers/opencv
rm -rf build
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D CUDA_ARCH_BIN=6.1 \
	-D WITH_CUBLAS=1 \
	-D OPENCV_EXTRA_MODULES_PATH=~/Installers/opencv_contrib/modules \
	-D HAVE_opencv_python3=ON \
	-D PYTHON_EXECUTABLE=~/.virtualenvs/baselines/bin/python \
	-D BUILD_EXAMPLES=OFF ..

make -j$(nproc)
sudo make install
sudo ldconfig
 sudo apt install jstest-gtk
  110  sudo pip install ds4drv
  111  sudo ds4drv
ls -l /usr/local/lib/python3.6/site-packages/cv2/python-3.6
cd ~/.virtualenvs/baselines/lib/python3.6/site-packages/
ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so

python3 -c "import cv2; count = cv2.cuda.getCudaEnabledDeviceCount(); print(count)" 

# Install baselines
sudo apt remove python3-dateutil -y
sudo apt update
sudo apt install python3-opencv -y
pip3 install  cloudpickle==1.3.0 torch==1.5.1 torchvision==0.6.1 seaborn
pip3 install imageio==2.6.1 tensorflow-gpu==1.15.2 keras==2.3.1 pyglet==1.3.0 keras-rl==0.4.2
pip3 install stable-baselines[mpi]==2.10.0  gym==0.17.2 stable-baselines3 optuna
pip3 uninstall opencv-python==4.2.0.34

# Install visual studio
sudo apt update
sudo apt install software-properties-common apt-transport-https wget
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt update
sudo apt install code

# Install tensorrt
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt6.0.1.8-ga-20191108_1-1_amd64.deb
sudo apt-key /var/add nv-tensorrt-repo-cuda10.2-trt6.0.1.8-ga-20191108/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt cuda-nvrtc-10-2
sudo apt-get install python3-libnvinfer-dev
dpkg -l | grep TensorRT
pip install albumentations==0.4.5 onnx==1.4.1
sudo find / -name tensorrt 2> /dev/null

# Install spinningup
sudo apt-get update && sudo apt-get install libopenmpi-dev
git clone https://github.com/openai/spinningup.git
cd spinningup
sudo apt-get install -y python-psutil
pip install -e .
python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999
