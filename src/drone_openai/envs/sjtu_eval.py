#! /usr/bin/env python

import gym
import numpy as np
import rospy

import utils.warning_ignore
from utils.saved_dir import model_dir, log_dir

import sjtu_goto
from stable_baselines.deepq import DQN, MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


def main():
    rospy.init_node('train_node', anonymous=True)
    env = gym.make("SJTUGotoEnv-v0")
    env = DummyVecEnv([lambda: env])
    model = PPO2.load(model_dir + "double_q", env=env)

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
