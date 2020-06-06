#!/bin/bash

source devel/setup.bash
# rosrun drone_openai sjtu_plot.py
rosrun drone_openai sjtu_train_dqn.py
# rosrun drone_openai sjtu_train_ppo2.py
# rosrun drone_openai sjtu_eval.py