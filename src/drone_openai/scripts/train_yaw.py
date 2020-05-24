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

import env_yaw
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

env = gym.make("Yaw-v0")
env = make_vec_env(lambda: env, n_envs=1)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

model = ACKTR('MlpPolicy', env, verbose=1)
model.learn(int(1e10), callback=eval_callback)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

model.save('models/acktr_yaw')

env.close()