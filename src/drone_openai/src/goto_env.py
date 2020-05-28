#! /usr/bin/env python

import rospy
import numpy
from gym import spaces
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion
import drone_env

reg = register(
        id='DroneGoto-v0',
        entry_point='drone_goto:DroneGotoEnv',
        max_episode_steps=100,
    )

class DroneGotoEnv(drone_env.DroneEnv):
    def __init__(self):
        return True