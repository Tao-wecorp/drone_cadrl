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

env = GoLeftEnv(grid_size=10)
env = make_vec_env(lambda: env, n_envs=1)
model = A2C('MlpPolicy', env, verbose=1).learn(500)

video_folder = 'videos/'
video_length = 100
prefix = 'a2c_goleft'
eval_env = GoLeftEnv(grid_size=10)
eval_env = make_vec_env(lambda: eval_env, n_envs=1)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
#                           record_video_trigger=lambda x: x == 0, video_length=video_length,
#                           name_prefix=prefix)

# obs = eval_env.reset()
# for _ in range(video_length-1):
#     action, _ = model.predict(obs)
#     obs, _, _, _ = eval_env.step(action)

eval_env.close()