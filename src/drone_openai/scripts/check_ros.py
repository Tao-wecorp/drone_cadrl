#! /usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

from stable_baselines3.common.env_checker import check_env
import env_ros
import gym
from gym.spaces import *

import rospy

from helpers.utils.gazebo_connection import GazeboConnection
from helpers.control import Control
control = Control()

class CheckEnv(gym.Env):
    def __init__(self):
        rospy.init_node('check_node', anonymous=True)
        self.gazebo = GazeboConnection()
        self.gazebo.resetSim()

        control.takeoff()
        rospy.on_shutdown(self.shutdown)

        env = gym.make("Yaw-v0")
        check_env(env, warn=True)
        obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print("Observation space:", env.observation_space)
        print("Shape:", env.observation_space.shape)
        print("Action space:", env.action_space)
        print("Sampled action:", action)
        print(obs.shape, reward, done, info)

    def shutdown(self):
        control.land()

def main():
    try:
        CheckEnv()
    except KeyboardInterrupt:
        pass
    
if __name__ == '__main__':
    main()

