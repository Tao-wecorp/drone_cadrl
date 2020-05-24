sudo apt-get install python-pip
sudo apt-get install ffmpeg freeglut3-dev xvfb
sudo apt-get install ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-joint-state-publisher-gui ros-melodic-teleop-twist-keyboard ffmpeg -y

pip3 install imageio==2.6.1 tensorflow-gpu==1.15.2 keras==2.3.1 pyglet==1.3.0 keras-rl==0.4.2
pip3 install stable-baselines[mpi]==2.10.0 torch==1.5.0 torchvision==0.6.0 gym==0.17.2 stable-baselines3 optuna
pip3 uninstall opencv-python==4.2.0.34

pip freeze > requirements.txt

echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/melodic/lib" >> ~/.bashrc
source ~/.bashrc

catkin_make -DPYTHON_EXECUTABLE:FILEPATH=~/.virtualenvs/py3venv/bin/python

