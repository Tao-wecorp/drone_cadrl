import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import parrotdrone_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion


reg = register(
        id='DroneGotoEnv-v0',
        entry_point='sjtu_goto:DroneGotoEnv',
    )

class DroneGotoEnv(parrotdrone_env.ParrotDroneEnv):
    def __init__(self):

        # Initial and Desired Point
        self.linear_forward_speed = 2
        self.init_angular_turn_speed = 0
        self.init_linear_speed_vector = Vector3()
        self.init_linear_speed_vector.x = 0
        self.init_linear_speed_vector.y = 0
        self.init_linear_speed_vector.z = 0
        
        self.desired_point = Point()
        self.desired_point.x = 5
        self.desired_point.y = 0
        self.desired_point.z = 0
        self.desired_point_epsilon = 0.5

        # Actions and Observations
        number_actions = 2
        self.action_space = spaces.Discrete(number_actions)

        self.work_space_x_max = 10
        self.work_space_x_min = -10
        self.work_space_y_max = 10
        self.work_space_y_min = -10
        self.work_space_z_max = 2
        self.work_space_z_min = 0
        high = numpy.array([self.work_space_x_max,
                            self.work_space_y_max,
                            self.work_space_z_max])
                                        
        low = numpy.array([ self.work_space_x_min,
                            self.work_space_y_min,
                            self.work_space_z_min])
        self.observation_space = spaces.Box(low, high)

        # Rewards
        self.reward_range = (-numpy.inf, numpy.inf)
        self.closer_to_point_reward = 0.1
        self.not_ending_point_reward = -0.01
        self.end_episode_points = 10
        self.cumulated_steps = 0.0

        super(DroneGotoEnv, self).__init__()

    def _set_init_pose(self):
        self.move_base(self.init_linear_speed_vector,
                    self.init_angular_turn_speed,
                    epsilon=0.05,
                    update_rate=30)
        return True

    def _init_env_variables(self):
        self.takeoff()
        self.cumulated_reward = 0.0
        gt_pose = self.get_gt_pose()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(gt_pose.position)

    def _set_action(self, action):
        rospy.logdebug("Action "+str(action))
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
                        update_rate=30)
    
    def _get_obs(self):
        gt_pose = self.get_gt_pose()                
       
        observations = numpy.array([round(gt_pose.position.x,1),
                        round(gt_pose.position.y,1),
                        round(gt_pose.position.z,1)])
        rospy.logdebug("Observations "+str(observations))
        return observations
        
    def _is_done(self, observations):       
        episode_done = False
        
        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]
        
        is_inside_workspace_now = self.is_inside_workspace(current_position)
        has_reached_des_point = self.is_in_desired_position(current_position, self.desired_point_epsilon)
        
        rospy.logwarn("RESULTS")
        if not is_inside_workspace_now:
            rospy.logerr("Out of workspace")
        
        if has_reached_des_point:
            rospy.logwarn("Reach target!")
        
        episode_done = not(is_inside_workspace_now) or has_reached_des_point

        if episode_done:
            rospy.logwarn("Episode done")

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
                rospy.logwarn("Rewarded")
                reward = self.closer_to_point_reward
            else:
                rospy.logerr("Punished")
                reward = 0
        else:
            if self.is_in_desired_position(current_position, epsilon=0.5):
                reward = self.end_episode_points
            else:
                reward = -1*self.end_episode_points


        self.previous_distance_from_des_point = distance_from_des_point

        rospy.logdebug("Reward: " + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward
    
    def is_in_desired_position(self, current_position, epsilon=0.5):
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
        
        rospy.logwarn("POSITION")
        rospy.logwarn("current position: " + str(current_position))
        
        return is_in_desired_pos
    
    def is_inside_workspace(self,current_position):
        is_inside = False

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                if current_position.z > self.work_space_z_min and current_position.z <= self.work_space_z_max:
                    is_inside = True
        
        return is_inside

    def get_distance_from_desired_point(self, current_position):
        distance = self.get_distance_from_point(current_position, self.desired_point)
    
        return distance

    def get_distance_from_point(self, pstart, p_end):
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))
        distance = numpy.linalg.norm(a - b)
    
        return distance