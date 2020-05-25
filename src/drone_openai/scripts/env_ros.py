#!/usr/bin/env python

import numpy as np
import rospy
import time

import gym
from gym import utils, spaces
from gym.utils import seeding
from gym.envs.registration import register

from geometry_msgs.msg import Twist, Vector3Stamped, Pose, Point
from sensor_msgs.msg import Imu
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

        self.speed_value = 1.0
        self.desired_pose = Pose()
        self.desired_pose.position.z = 0
        self.desired_pose.position.x = 5
        self.desired_pose.position.y = 0
        
        self.running_step = 2
        self.max_incl = 0.7
        self.max_altitude = 2.0

        self.gazebo = GazeboConnection()

        self.grid_size = grid_size
        self.agent_pos = grid_size - 1

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def take_observation (self):
        data_pose = None
        while data_pose is None:
            try:
                data_pose = rospy.wait_for_message('/drone/gt_pose', Pose, timeout=5)
            except:
                rospy.loginfo("Current drone pose not ready yet, retrying for getting robot pose")
        return data_pose

    def reset(self):
        self.gazebo.resetSim()
        self.observation = np.array([0]).astype(np.float32)

        return self.observation

    def step(self, action):

        vel_cmd = Twist()
        if action == 0: #FORWARD
            vel_cmd.linear.x = self.speed_value
            vel_cmd.angular.z = 0.0
        elif action == 1: #LEFT
            vel_cmd.linear.x = 1
            vel_cmd.angular.z = self.speed_value
        elif action == 2: #RIGHT
            vel_cmd.linear.x = 1
            vel_cmd.angular.z = -self.speed_value
        elif action == 3: #Up
            vel_cmd.linear.z = self.speed_value
            vel_cmd.angular.z = 0.0
        elif action == 4: #Down
            vel_cmd.linear.z = -self.speed_value
            vel_cmd.angular.z = 0.0
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        self.vel_pub.publish(vel_cmd)
        time.sleep(self.running_step)
        data_pose = self.take_observation()
        self.observation = np.array([data_pose.position.x]).astype(np.float32)

        reward = 1- ((5 - data_pose.position.x)/5)
        done = bool(reward >= 0.9)
        info = {}

        return self.observation, reward, done, info

    def render(self):
        pass

    def close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]