#! /usr/bin/env python

import gym
import numpy as np

import os
import rospy
import rospkg
import utils.warning_ignore
rospack = rospkg.RosPack()
model_folder = os.path.join(rospack.get_path("drone_openai"), "envs/models/")

import sjtu_goto
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_vec_env


def main():
    rospy.init_node('train_node', anonymous=True)
    num_cpu = 4
    env = make_vec_env("SJTUGotoEnv-v0", n_envs=num_cpu)
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10)

    env_val = gym.make("SJTUGotoEnv-v0")
    mean_reward, std_reward = evaluate_policy(model, env_val, n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    model.save(model_folder + "double_q")

if __name__ == '__main__':
    main()