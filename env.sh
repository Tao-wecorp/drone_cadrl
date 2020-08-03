# nvidia drivers
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# cuda 10.2
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*
sudo apt update
sudo add-apt-repository ppa:graphics-drivers
sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
sudo apt update
sudo apt install cuda-10-2
sudo apt install libcudnn7
sudo nano ~/.profile
# set PATH for cuda 10.1 installation
if [ -d "/usr/local/cuda-10.1/bin/" ]; then
    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

# ros
sudo apt-get install python-pip
sudo apt-get install ffmpeg freeglut3-dev xvfb
sudo apt-get install ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-joint-state-publisher-gui ros-melodic-teleop-twist-keyboard ffmpeg -y

# torch
pip3 install torch==1.5.1 torchvision==0.6.1
# tensorflow
pip3 install imageio==2.6.1 tensorflow-gpu==1.15.2 keras==2.3.1 pyglet==1.3.0 keras-rl==0.4.2
# baselines
pip3 install stable-baselines[mpi]==2.10.0  gym==0.17.2 stable-baselines3 optuna
pip3 uninstall opencv-python==4.2.0.34
# osnet
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid/
pip install -r requirements.txt
pip3 uninstall opencv-python==4.3.6.0
python setup.py develop

pip freeze > requirements.txt

echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/melodic/lib" >> ~/.bashrc
source ~/.bashrc

catkin_make -DPYTHON_EXECUTABLE:FILEPATH=~/.virtualenvs/py3venv/bin/python

