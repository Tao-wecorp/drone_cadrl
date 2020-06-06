#! /usr/bin/env python

import gym
import numpy as np
import rospy

import utils.warning_ignore
from utils.saved_dir import model_dir, log_dir

import sjtu_goto
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.bench import Monitor


def main():
    rospy.init_node('train_node', anonymous=True)
    env = gym.make("SJTUGotoEnv-v0")
    # env = DummyVecEnv([lambda: env])
    env = Monitor(env, log_dir)
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    model.save(model_dir + "double_q")

if __name__ == '__main__':
    main()