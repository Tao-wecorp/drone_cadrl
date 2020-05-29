#! /usr/bin/env python

import gym
import numpy as np

from stable_baselines.deepq import DQN, MlpPolicy

import flyto_env

def callback(lcl, _glb):
    if len(lcl['episode_rewards'][-101:-1]) == 0:
        mean_100ep_reward = -np.inf
    else:
        mean_100ep_reward = round(float(np.mean(lcl['episode_rewards'][-101:-1])), 1)
    is_solved = lcl['self'].num_timesteps > 100 and mean_100ep_reward >= 199
    return not is_solved


def main():
    env = gym.make("DroneFlyto-v0")
    model = DQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
    )
    model.learn(total_timesteps=1000, callback=callback)

    # model.save("cartpole_model.zip")


if __name__ == '__main__':
    main()
