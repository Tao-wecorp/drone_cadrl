#!/usr/bin/env python

import gym
import rospy
import time
import numpy as np

from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty as EmptyTopicMsg
from gym.utils import seeding
from gym.envs.registration import register
from helpers.utils.gazebo_connection import GazeboConnection


reg = register(
  id='Yaw-v0',
  entry_point='env_yaw:YawEnv',
  max_episode_steps=100,
  )


class YawEnv(gym.Env):
  
  LEFT = 0
  RIGHT = 1

  def __init__(self, grid_size=10):
    super(YawEnv, self).__init__()

    self.grid_size = grid_size
    self.agent_pos = grid_size - 1

    n_actions = 2
    self.action_space = spaces.Discrete(n_actions)
    self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(1,), dtype=np.float64)

  def reset(self):
    self.agent_pos = self.grid_size - 1
    return np.array([self.agent_pos]).astype(np.float32)

  def step(self, action):
    if action == self.LEFT:
      self.agent_pos -= 1
    elif action == self.RIGHT:
      self.agent_pos += 1
    else:
      raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

    done = bool(self.agent_pos == 0)

    reward = 1 if self.agent_pos == 0 else 0

    info = {}

    return np.array([self.agent_pos]).astype(np.float32), reward, done, info

  def render(self):
    print("." * self.agent_pos, end="")
    print("x", end="")
    print("." * (self.grid_size - self.agent_pos))

  def close(self):
    pass