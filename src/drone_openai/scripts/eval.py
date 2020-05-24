#! /usr/bin/env python

import gym
gym.logger.set_level(40)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from env import GoLeftEnv
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy


eval_env = GoLeftEnv(grid_size=10)
eval_env = make_vec_env(lambda: eval_env, n_envs=1)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")