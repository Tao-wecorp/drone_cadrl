#! /usr/bin/env python

import gym
gym.logger.set_level(40)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from env import GoLeftEnv
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


env = GoLeftEnv(grid_size=10)
env = make_vec_env(lambda: env, n_envs=1)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

model = ACKTR('MlpPolicy', env, verbose=1)
model.learn(int(1e10), callback=eval_callback)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

model.save('models/best')

env.close()