#! /usr/bin/env python

import gym
import numpy as np
import rospy
import utils.warning_ignore

import sjtu_goto
from stable_baselines.deepq import DQN, MlpPolicy


def main():
    rospy.init_node('train_node', anonymous=True)
    env = gym.make("SJTUGotoEnv-v0")
    model = DQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
    )
    model.learn(total_timesteps=1000)


if __name__ == '__main__':
    main()
