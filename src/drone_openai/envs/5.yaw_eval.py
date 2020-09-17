#! /usr/bin/env python

import gym
import numpy as np

import matplotlib.pyplot as plt

import rospy

import utils.warning_ignore
from utils.saved_dir import model_dir, log_dir

import yaw_task
from stable_baselines.deepq import DQN, MlpPolicy
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


def main():
    rospy.init_node('eval_node', anonymous=True)
    env = gym.make("SJTUYawEnv-v0")
    env = DummyVecEnv([lambda: env])
    model = PPO2.load(model_dir + "yaw_ppo2", env=env)

    obs = env.reset()
    n_steps = 20
    for step in range(n_steps):
        action, _states = model.predict(obs)
        print("Step {}".format(step + 1))
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        if done:
            print("Goal reached!", "reward=", reward)
            break

if __name__ == '__main__':
    main()
