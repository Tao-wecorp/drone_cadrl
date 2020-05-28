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
        id='DroneFlyto-v0',
        entry_point='flyto_env:DroneFlytoEnv',
        max_episode_steps=1000,
    )

class DroneFlytoEnv(drone_env.DroneEnv):
    def __init__(self):
        return True