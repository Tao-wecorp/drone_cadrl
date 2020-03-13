# Jetbot CADRL
Running CADRL_ROS on Jetbot with Gazebo simulation

## To-do
1. Import Jetbot model into Gazebo;
2. Train DQN single agent model;
3. Train CADRL multi-agent model;
4. Deploy models onto Jebot. 

## Env
    sudo apt-get install python-pip ros-melodic-ros-control ros-melodic-ros-controllers
    pip2 install imageio==2.6.1
    pip2 install tensorflow-gpu==1.14.0 keras==2.3.1
    pip2 install cvlib --no-deps
    pip2 install requests progressbar imutils  opencv-python