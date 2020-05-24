#! /usr/bin/env python

import gym
gym.logger.set_level(40)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from env import GoLeftEnv
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env



env = GoLeftEnv(grid_size=10)
env = make_vec_env(lambda: env, n_envs=1)
model = ACKTR('MlpPolicy', env, verbose=1).learn(5000)