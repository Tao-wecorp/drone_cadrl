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
        max_episode_steps=1000,
    )

class DroneGotoEnv(drone_env.DroneEnv):
    def __init__(self):
        
        # Init Speed
        self.linear_forward_speed = 10
        self.init_linear_speed_vector = Vector3()
        self.init_linear_speed_vector.x = 0
        self.init_linear_speed_vector.y = 0
        self.init_linear_speed_vector.z = 0
        self.init_angular_turn_speed = 0

        # Desired Point
        self.desired_point = Point()
        self.desired_point.x = 5
        self.desired_point.y = 0
        self.desired_point.z = 0
        self.desired_point_epsilon = 0.1

        # WorkSpace Cube Dimensions
        self.work_space_x_max = 0
        self.work_space_x_min = 10
        self.work_space_y_max = 5
        self.work_space_y_min = -5
        self.work_space_z_max = 0
        self.work_space_z_min = 2
        
        # Maximum RPY values
        self.max_roll = 45
        self.max_pitch = 45
        self.max_yaw = 45

        high = numpy.array([self.work_space_x_max,
                            self.work_space_y_max,
                            self.work_space_z_max,
                            self.max_roll,
                            self.max_pitch,
                            self.max_yaw])
                                            
        low = numpy.array([ self.work_space_x_min,
                            self.work_space_y_min,
                            self.work_space_z_min,
                            -1*self.max_roll,
                            -1*self.max_pitch,
                            -1*self.max_yaw])
        
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(4)
        self.reward_range = (-numpy.inf, numpy.inf)

        self.closer_to_point_reward = 10
        self.not_ending_point_reward = 10
        self.end_episode_points = 100

        self.cumulated_steps = 0.0

        super(DroneGotoEnv, self).__init__()

    def reset(self):
        rospy.logdebug("Reseting RobotGazeboEnvironment")
        self._reset_sim()
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        rospy.logdebug("END Reseting RobotGazeboEnvironment")
        return obs

    def _set_init_pose(self):
        self.move_base(self.init_linear_speed_vector,
                        self.init_angular_turn_speed,
                        epsilon=0.05,
                        update_rate=10)
        # self.land()
        return True

    def _init_env_variables(self):
        self.takeoff()
        self.cumulated_reward = 0.0
        gt_pose = self.get_gt_pose()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(gt_pose.position)

        
    def _set_action(self, action):
        linear_speed_vector = Vector3()
        angular_speed = 0.0
        
        if action == 0: #FORWARDS
            linear_speed_vector.x = self.linear_forward_speed
            self.last_action = "FORWARDS"
        elif action == 1: #BACKWARDS
            linear_speed_vector.x = -1*self.linear_forward_speed
            self.last_action = "BACKWARDS"
        elif action == 2: #STRAFE_LEFT
            linear_speed_vector.y = self.linear_forward_speed
            self.last_action = "STRAFE_LEFT"
        elif action == 3: #STRAFE_RIGHT
            linear_speed_vector.y = -1*self.linear_forward_speed
            self.last_action = "STRAFE_RIGHT"
        elif action == 4: #UP
            linear_speed_vector.z = self.linear_forward_speed
            self.last_action = "UP"
        elif action == 5: #DOWN
            linear_speed_vector.z = -1*self.linear_forward_speed
            self.last_action = "DOWN"

        self.move_base(linear_speed_vector,
                        angular_speed,
                        epsilon=0.05,
                        update_rate=10)

    def _get_obs(self):
        gt_pose = self.get_gt_pose()

        roll, pitch, yaw = self.get_orientation_euler(gt_pose.orientation)

        observations = [int(gt_pose.position.x),
                        int(gt_pose.position.y),
                        int(gt_pose.position.z),
                        round(roll,1),
                        round(pitch,1),
                        round(yaw,1)]
        return numpy.array(observations)
    
    def _is_done(self, observations):
        episode_done = False
        
        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]
        
        current_orientation = Point()
        current_orientation.x = observations[3]
        current_orientation.y = observations[4]
        current_orientation.z = observations[5]

        is_inside_workspace_now = self.is_inside_workspace(current_position)
        has_reached_des_point = self.is_in_desired_position(current_position, self.desired_point_epsilon)

        episode_done = not(is_inside_workspace_now) or has_reached_des_point

        return episode_done
    
    def _compute_reward(self, observations, done):

        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference =  distance_from_des_point - self.previous_distance_from_des_point


        if not done:
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward = self.closer_to_point_reward
            else:
                rospy.logerr("ENCREASE IN DISTANCE BAD")
                reward = 0
                
        else:
            if self.is_in_desired_position(current_position, epsilon=0.5):
                reward = self.end_episode_points
            else:
                reward = -1*self.end_episode_points


        self.previous_distance_from_des_point = distance_from_des_point
        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        return reward
    
    def is_in_desired_position(self,current_position, epsilon=0.05): 
        is_in_desired_pos = False
           
        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon
        
        x_current = current_position.x
        y_current = current_position.y
        
        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)
        
        is_in_desired_pos = x_pos_are_close and y_pos_are_close
        
        return is_in_desired_pos
    
    def is_inside_workspace(self,current_position):
        is_inside = False

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                if current_position.z > self.work_space_z_min and current_position.z <= self.work_space_z_max:
                    is_inside = True
        
        return is_inside
        
    def get_distance_from_desired_point(self, current_position):
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)
    
        return distance
    
    def get_distance_from_point(self, pstart, p_end):
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))
    
        distance = numpy.linalg.norm(a - b)
    
        return distance
        
    def get_orientation_euler(self, quaternion_vector):
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]
    
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw


    