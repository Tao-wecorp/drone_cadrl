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
from stable_baselines.deepq import DQN, MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def main():
    rospy.init_node('train_node', anonymous=True)
    env = gym.make("SJTUGotoEnv-v0")

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

    kwargs = {'double_q': True, 'prioritized_replay': True, 'policy_kwargs': dict(dueling=True)}
    model = DQN(
        env=env,
        policy=MlpPolicy, 
        **kwargs
    )
    model.learn(int(1e10), callback=eval_callback)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    model.save(model_folder + "double_q")

if __name__ == '__main__':
    main()
