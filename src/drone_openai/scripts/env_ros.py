#!/usr/bin/env python

import numpy as np
import rospy
import time

import gym
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register

from geometry_msgs.msg import Twist, Pose, Point
from std_msgs.msg import Empty as EmptyTopicMsg, Bool

from helpers.utils.gazebo_connection import GazeboConnection
from helpers.control import Control
control = Control()

reg = register(
    id='Yaw-v0',
    entry_point='env_ros:YawEnv',
    max_episode_steps=100,
    )


class YawEnv(gym.Env):

    def __init__(self, grid_size=10):
        super(YawEnv, self).__init__()

        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

        self.speed = 2.0
        self.running_time = 0.5

        self.desired_pose = Pose()
        self.desired_pose.position.x = 5

        self.gazebo = GazeboConnection()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def observe(self):
        pose = None
        while pose is None:
            try:
                pose = rospy.wait_for_message('/drone/gt_pose', Pose, timeout=5)
            except:
                rospy.loginfo("Current drone pose not ready yet, retrying for getting robot pose")
        return pose

    def reset(self):
        self.gazebo.resetSim()
        self.observation = np.array([0]).astype(np.float32)

        return self.observation

    def step(self, action):

        vel_cmd = Twist()
        if action == 0: #FORWARD
            vel_cmd.linear.x = self.speed
        elif action == 1: #BACKWARD
            vel_cmd.linear.x = -self.speed
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        self.vel_pub.publish(vel_cmd)
        time.sleep(self.running_time)
        pose = self.observe()
        observation = np.array([pose.position.x]).astype(np.float32)
        print(observation)

        reward = 1- ((5 - pose.position.x)/5)
        done = bool(reward >= 0.9)
        info = {}

        return observation, reward, done, info

    def render(self):
        pass

    def close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]