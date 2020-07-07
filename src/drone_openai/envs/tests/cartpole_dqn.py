#! /usr/bin/env python

import gym
import numpy as np

from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy

import utils.warning_ignore
from utils.saved_dir import model_dir, log_dir

def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


kwargs = {'double_q': True, 'prioritized_replay': True, 'policy_kwargs': dict(dueling=True)}
model = DQN('MlpPolicy', 'CartPole-v1', verbose=0, **kwargs)
mean_reward_before_train = evaluate(model, num_episodes=100)

model = model = DQN('MlpPolicy', 'CartPole-v1', verbose=0, **kwargs).learn(1000)
mean_reward_before_train = evaluate(model, num_episodes=100)
