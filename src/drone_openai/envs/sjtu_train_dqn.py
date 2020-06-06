#! /usr/bin/env python

import gym
import numpy as np
import rospy

import utils.warning_ignore
from utils.saved_dir import model_dir, log_dir

import sjtu_goto
from stable_baselines.deepq import DQN, MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def main():
    rospy.init_node('train_node', anonymous=True)
    env = gym.make("SJTUGotoEnv-v0")
    env = Monitor(env, log_dir)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1, verbose=1)
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

    model.save(model_dir + "double_q")

if __name__ == '__main__':
    main()
