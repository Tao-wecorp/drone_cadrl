#! /usr/bin/env python

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from stable_baselines.bench import Monitor
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import warning_ignore
from saved_dir import model_dir, log_dir

env = gym.make('CartPole-v1')
# env = DummyVecEnv([lambda: env])
env = Monitor(env, log_dir)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=50000, callback=eval_callback, reset_num_timesteps=False)

model.save(model_dir + "cartpole_ppo2")
