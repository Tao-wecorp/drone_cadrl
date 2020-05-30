#! /usr/bin/env python

import rospy
import numpy as np
import time
from gym import spaces
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion
from openai_ros.robot_envs import parrotdrone_env

reg = register(
        id='DroneGoto-v0',
        entry_point='drone_goto:DroneGotoEnv',
    )


class DroneGotoEnv(parrotdrone_env.ParrotDroneEnv):
    def __init__(self):
        
        self.linear_forward_speed = 10

        self.action_space = spaces.Discrete(2)

        high = np.array([5,5,2])                              
        low = np.array([-5,-5,0])
        self.observation_space = spaces.Box(low, high)
        self.reward_range = (-np.inf, np.inf)
        self.reward = 0

        super(DroneGotoEnv, self).__init__()
        

    def _set_init_pose(self):

        linear_speed_vector = Vector3()
        linear_speed_vector.x = 0
        linear_speed_vector.y = 0
        linear_speed_vector.z = 0

        self.move_base(linear_speed_vector,
                        0,
                        epsilon=0.05,
                        update_rate=10)
        self.land()
        return True

    def _init_env_variables(self):
        self.takeoff()
        self.cumulated_steps = 0.0
        self.cumulated_reward = 0.0
        gt_pose = self._get_obs()

    def _set_action(self, action):
        linear_speed_vector = Vector3()
        angular_speed = 0.0
        
        if action == 0: #FORWARDS
            linear_speed_vector.x = self.linear_forward_speed
            self.last_action = "FORWARDS"
        elif action == 1: #BACKWARDS
            linear_speed_vector.x = -1*self.linear_forward_speed
            self.last_action = "BACKWARDS"


        self.move_base(linear_speed_vector,
                        angular_speed,
                        epsilon=0.05,
                        update_rate=10)
        time.sleep(0.1)

    def _get_obs(self):
        gt_pose = self.get_gt_pose()

        observations = np.array([int(gt_pose.position.x),
                            int(gt_pose.position.y),
                            int(gt_pose.position.z)])
        return observations
    
    def _is_done(self, observations):
        episode_done = False
        episode_done = bool(self.reward >= 0.98)
        if episode_done:
            print("Done!")
        return episode_done
    
    def _compute_reward(self, observations, done):
        current_position = Point()
        current_position.x = observations[0]
        self.reward = 1- ((5 - current_position.x)/5)
        self.cumulated_reward += self.reward
        self.cumulated_steps += 1
        
        return self.reward



    