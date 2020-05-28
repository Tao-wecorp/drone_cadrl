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

import env_ros
import gym
import rospy

from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


class TrainEnv(gym.Env):
    def __init__(self):
        rospy.init_node('train_node', anonymous=True)

        env = gym.make("Yaw-v0")
        env = make_vec_env(lambda: env, n_envs=1)
        model = DQN('MlpPolicy', env, verbose=1).learn(1000)

def main():
    try:
        TrainEnv()
    except KeyboardInterrupt:
        pass
    
if __name__ == '__main__':
    main()