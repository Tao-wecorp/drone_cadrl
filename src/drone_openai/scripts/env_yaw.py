#!/usr/bin/env python

import gym
import rospy
import time
import numpy as np

from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose, Point
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty as EmptyTopicMsg, Bool
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

        self.pos_pub = rospy.Publisher('/drone/cmd_pos',Point,queue_size=1)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.takeoff_pub = rospy.Publisher('/drone/takeoff', EmptyTopicMsg, queue_size=0)
        self.switch_pub = rospy.Publisher('/drone/posctrl',Bool,queue_size=1)

        self.desired_pose = Pose()
        self.desired_pose.position.z = 0
        self.desired_pose.position.x = 5
        self.desired_pose.position.y = 5
        
        self.running_step = 2
        self.max_incl = 0.3
        self.minx = -5
        self.maxx = 5
        self.miny = -5
        self.maxy = 5
        self.shape = (10,10)

        self.incx = (self.maxx - self.minx)/ (self.shape[0])
        self.incy = (self.maxy - self.miny)/ (self.shape[1])
        self.horizontal_bins = np.zeros((2,self.shape[1]))
        
        self.horizontal_bins[0] = np.linspace(self.minx,self.maxx,self.shape[1])
        self.horizontal_bins[1] = np.linspace(self.miny,self.maxy,self.shape[1])
        self.goal = np.zeros(2)
        self.goal[0] = int(np.digitize(self.desired_pose.position.x,self.horizontal_bins[0]))
        self.goal[1] = int(np.digitize(self.desired_pose.position.y,self.horizontal_bins[1]))
        print("Goal: %s"%(self.goal))
        self.gazebo = GazeboConnection()
        
        self.action_space = spaces.Discrete(4) #Forward,Backward,Left,Right
        self.reward_range = (-np.inf, np.inf)

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