#! /usr/bin/env python

import gym
import numpy as np
import matplotlib.pyplot as plt
import rospy

import utils.warning_ignore
from utils.saved_dir import model_dir, log_dir

import parrotdrone_goto
from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def main():
    rospy.init_node('train_node', anonymous=True)
    env_id = "ParrotDroneGoto-v0"
    num_cpu = 36

    env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=20000, reset_num_timesteps=False)

    model.save(model_dir + "goto_ppo2")

if __name__ == '__main__':
    main()