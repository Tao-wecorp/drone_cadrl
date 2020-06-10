#! /usr/bin/env python

import rospy
import numpy
from math import *

from gym import spaces
from gym.envs.registration import register
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import yaw_env

reg = register(
        id='SJTUYawEnv-v0',
        entry_point='yaw_task:SJTUYawEnv',
    )


class SJTUYawEnv(yaw_env.SJTUDroneEnv):
    def __init__(self):

        # Initial and Desired Point
        self.linear_forward_speed = 0
        self.angular_turn_speed = 1

        self.init_angular_turn_speed = 0
        self.init_linear_speed_vector = Vector3()
        self.init_linear_speed_vector.x = 0
        self.init_linear_speed_vector.y = 0
        self.init_linear_speed_vector.z = 0

        self.desired_orientation = Point()
        self.desired_orientation.x = 0
        self.desired_orientation.y = 0
        self.desired_orientation.z = 5

        # Actions and Observations
        number_actions = 2
        self.action_space = spaces.Discrete(number_actions)

        self.max_yaw = 1.57
        high = numpy.array([self.max_yaw])                                       
        low = numpy.array([-1*self.max_yaw])
        self.observation_space = spaces.Box(low, high)

        # Rewards
        self.reward_range = (-numpy.inf, numpy.inf)
        self.closer_to_pose_reward = 0.1
        self.end_episode_points = 10
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
    
    def _set_action(self, action):
        rospy.logdebug("Action "+str(action))
        angular_speed = 0.0
        
        if action == 0: #Yaw Left
            angular_speed = self.angular_turn_speed
        elif action == 1: #Yaw Right
            angular_speed = -1*self.angular_turn_speed

        self.move_base(self.init_linear_speed_vector,
                        angular_speed,
                        epsilon=0.05,
                        update_rate=10)

    def _get_obs(self):
        gt_pose = self.get_gt_pose()

        roll, pitch, yaw = self.get_orientation_euler(gt_pose.orientation)
              
        observations = numpy.array([round(yaw,1)])
        rospy.logdebug("Observations "+str(observations))
        return observations

    def _is_done(self, observations):
        episode_done = False

        print(observations)

        yaw = observations[0]

        is_inside_workspace_now = self.is_inside_workspace(yaw)
        has_reached_des_point = self.is_in_desired_position(yaw)

        episode_done = bool(not(is_inside_workspace_now) or has_reached_des_point)

        return episode_done

    def _compute_reward(self, observations, done):

        yaw = observations[0]

        if self.is_in_desired_position(yaw):
            reward = self.end_episode_points
        else:
            reward = -1*self.end_episode_points

        self.cumulated_reward += reward
        self.cumulated_steps += 1

        return reward

    
    def is_in_desired_position(self, yaw_current, epsilon=0.5):
        is_in_desired_pos = False

        yaw_plus = self.desired_orientation.z + epsilon
        yaw_minus = self.desired_orientation.z - epsilon

        is_in_desired_pos = bool((yaw_current <= yaw_plus) and (yaw_current > yaw_minus))

        return is_in_desired_pos

    def is_inside_workspace(self, yaw):
        is_inside = False

        is_inside = bool((yaw > -1*self.max_yaw) and (yaw <= self.max_yaw))

        return is_inside

    def get_orientation_euler(self, quaternion_vector):
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]
    
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw