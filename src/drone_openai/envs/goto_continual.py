#! /usr/bin/env python

import gym
import numpy as np
import matplotlib.pyplot as plt
import rospy

import utils.warning_ignore
from utils.saved_dir import model_dir, log_dir

import parrotdrone_goto
from stable_baselines import PPO2
from stable_baselines.deepq import DQN, MlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

def main():
    rospy.init_node('train_node', anonymous=True)
    env = gym.make("ParrotDroneGoto-v0")
    env = DummyVecEnv([lambda: env])

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

    model = PPO2.load(model_dir + "goto_dqn", tensorboard_log=log_dir)
    model.set_env(env)
    model.learn(total_timesteps=10000, callback=eval_callback, reset_num_timesteps=False)

    model.save(model_dir + "goto_dqn_20000")

if __name__ == '__main__':
    main()