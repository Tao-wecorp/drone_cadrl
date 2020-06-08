#! /usr/bin/env python

import gym
import numpy as np
import matplotlib.pyplot as plt
import rospy

import utils.warning_ignore
from utils.saved_dir import model_dir, log_dir

import sjtu_goto
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def main():
    rospy.init_node('train_node', anonymous=True)
    env = gym.make("SJTUGotoEnv-v0")
    env = Monitor(env, log_dir)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=5000, callback=eval_callback)

    model.save(model_dir + "ppo2")

if __name__ == '__main__':
    main()