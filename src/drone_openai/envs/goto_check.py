#! /usr/bin/env python

import gym
from gym.spaces import *
import rospy

import utils.warning_ignore
from utils.saved_dir import model_dir, log_dir

import parrotdrone_goto
from stable_baselines.common.env_checker import check_env


class CheckEnv(parrotdrone_goto.ParrotDroneGotoEnv):
    def __init__(self):
        rospy.init_node('check_node', anonymous=True)

        env = gym.make("ParrotDroneGoto-v0")
        obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print("Observation space:", env.observation_space)
        print("Shape:", env.observation_space.shape)
        print("Action space:", env.action_space)
        print("Sampled action:", action)

def main():
    try:
        CheckEnv()
    except KeyboardInterrupt:
        pass
    
if __name__ == '__main__':
    main()