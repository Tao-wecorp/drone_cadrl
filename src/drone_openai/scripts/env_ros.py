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
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    BACKWARD = 3
    ACTIONS = [LEFT,RIGHT,FORWARD,BACKWARD]

    def __init__(self, grid_size=10):
        super(YawEnv, self).__init__()

        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

        self.desired_pose = Pose()
        self.desired_pose.position.z = 0
        self.desired_pose.position.x = 5
        self.desired_pose.position.y = 0
        
        self.running_step = 2
        self.max_incl = 0.3
        self.minx = -5
        self.maxx = 5
        self.miny = -5
        self.maxy = 5
        self.shape = (10,10)


        self.horizontal_bins = np.zeros((2,self.shape[1]))
        
        self.horizontal_bins[0] = np.linspace(self.minx,self.maxx,self.shape[1])
        self.horizontal_bins[1] = np.linspace(self.miny,self.maxy,self.shape[1])
        self.goal = np.zeros(2)
        self.goal[0] = int(np.digitize(self.desired_pose.position.x,self.horizontal_bins[0]))
        self.goal[1] = int(np.digitize(self.desired_pose.position.y,self.horizontal_bins[1]))
        print("Goal: %s"%(self.goal))

        self.gazebo = GazeboConnection()

        self.grid_size = grid_size
        self.agent_pos = grid_size - 1

        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(1,), dtype=np.float64)
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

    def init_desired_pose(self):
        current_init_pose = self.take_observation()
        self.best_dist = current_init_pose.position.x - self.desired_pose.position.x

    def reset(self):
        self.gazebo.resetSim()
        self.gazebo.unpauseSim()

        self.init_desired_pose()
        self.agent_pos = 0

        self.gazebo.pauseSim()

        return np.array([self.agent_pos]).astype(np.float32)

    def step(self, action):
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        elif action == self.FORWARD:
            self.agent_pos -= 0
        elif action == self.BACKWARD:
            self.agent_pos += 0
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        done = bool(self.agent_pos == 0)

        reward = 1 if self.agent_pos == 0 else 0

        info = {}

        return np.array([self.agent_pos]).astype(np.float32), reward, done, info

    def render(self):
        pass

    def close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]