#! /usr/bin/env python

import rospy
import numpy
from math import *

from gym import spaces
from gym.envs.registration import register
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import sjtu_env

reg = register(
        id='SJTUYawEnv-v0',
        entry_point='yaw_task:SJTUYawEnv',
    )


class SJTUYawEnv(sjtu_env.SJTUDroneEnv):
    def __init__(self):

        # Initial and Desired Point
        self.angular_turn_speed = 1
        self.init_angular_turn_speed = 0
        self.init_linear_speed_vector = Vector3()
        self.init_linear_speed_vector.x = 0
        self.init_linear_speed_vector.y = 0
        self.init_linear_speed_vector.z = 0
        
        yaw_angle = 45
        self.desired_pose = Pose()
        self.desired_pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, yaw_angle*pi/180))

        # Actions and Observations
        number_actions = 2
        self.action_space = spaces.Discrete(number_actions)

        self.work_space_angle_max = 90
        self.work_space_angle_min = -90
        high = numpy.array([self.work_space_angle_max])                                       
        low = numpy.array([self.work_space_angle_min])
        self.observation_space = spaces.Box(low, high)

        # Rewards
        self.reward_range = (-numpy.inf, numpy.inf)
        self.closer_to_pose_reward = 0.1
        self.end_episode_pose = 10
        self.cumulated_steps = 0.0

        super(SJTUYawEnv, self).__init__()
    
    def _set_init_pose(self):
        self.move_base(self.init_linear_speed_vector,
                    self.init_angular_turn_speed,
                    epsilon=0.05,
                    update_rate=10)
        return True

    def _init_env_variables(self):
        self.takeoff()
        self.cumulated_reward = 0.0
        gt_pose = self.get_gt_pose()

def main():
    try:
        SJTUYawEnv()
    except KeyboardInterrupt:
        pass
    
if __name__ == '__main__':
    main()