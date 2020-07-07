#! /usr/bin/env python

import gym
import numpy as np

from stable_baselines import PPO2
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


env = gym.make('CartPole-v1')
model = PPO2(MlpPolicy, env, verbose=0)
mean_reward_before_train = evaluate(model, num_episodes=100)

model = PPO2('MlpPolicy', "CartPole-v1", verbose=0).learn(1000)
model.save(model_dir + "/ppo2")
mean_reward_before_train = evaluate(model, num_episodes=100)

obs = model.env.observation_space.sample()
print("pre saved", model.predict(obs, deterministic=True))
del model

loaded_model = PPO2.load(model_dir + "ppo2")
print("loaded", loaded_model.predict(obs, deterministic=True))
