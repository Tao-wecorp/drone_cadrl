import gym
gym.logger.set_level(40)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines3.common.env_checker import check_env
from env import GoLeftEnv
env = GoLeftEnv()

check_env(env, warn=True)

print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
print("Action space:", env.action_space)

obs = env.reset()
action = env.action_space.sample()
print("Sampled action:", action)
obs, reward, done, info = env.step(action)
print(obs, reward, done, info)