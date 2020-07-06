#! /usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_srvs.srv import Empty

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Point, Pose, Quaternion, Twist

from copy import deepcopy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from cvlib.object_detection import draw_bbox

from math import *
import numpy as np
import time

# Helpers
from helpers.cvlib import Detection
detection = Detection()

from helpers.control import Control
control = Control()
pose = Pose()
fpv = [320, 480]

from helpers.utils.gazebo_connection import GazeboConnection
gazebo = GazeboConnection()

class Yaw(object):
    def __init__(self):
        # Init
        rospy.init_node('yaw_node', anonymous=True)
        self.rate = rospy.Rate(30)
        self.frame = None
        self.bridge_object = CvBridge()
        
        self.robot_position = None

        # Sub & Pub
        rospy.Subscriber("/drone/front_camera/image_raw",Image,self.cam_callback)
        self.states_sub = rospy.Subscriber("/gazebo/model_states",ModelStates,self.states_callback)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Takeoff & Land
        control.takeoff()
        rospy.on_shutdown(self.shutdown)

        # Create centroids detection thread
        self.cent_thr = Thread(target=self.detect_cent, args=())
        self.cent_thr.daemon = True
        self.cent_thr.start()

        # Yaw
        yaw_angle = 0
        state_robot_msg = ModelState()
        state_robot_msg.model_name = 'sjtu_drone'
        while not rospy.is_shutdown():
            if self.frame is not None:
                start_time = time.time()
                frame = deepcopy(self.frame)
                robot_position = deepcopy(self.robot_position)
                
                
                yaw_angle = yaw_angle + 1
                rospy.wait_for_service('/gazebo/set_model_state')
                try:
                    pose.position = robot_position
                    pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, yaw_angle*pi/180))
                    state_robot_msg.pose = pose

                    self.set_state(state_robot_msg)
                except rospy.ServiceException:
                    pass
                 
                print("%s seconds" % (time.time() - start_time))
                # time.sleep((time.time() - start_time))
                
            self.rate.sleep()
    
    # Methods
    def cam_callback(self,data):
        try:
            cv_img = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        self.frame = cv_img

    def detect_cent(self):
        r = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.frame is not None:
                frame = deepcopy(self.frame)

                start_time = time.time()
                centroids = detection.detect(frame)
                # print("%s seconds" % (time.time() - start_time))

                # To-do: multithread or action node
                if len(centroids)==0: 
                    continue
                else:
                    cent = centroids[0]
                    self.yaw_angle = openpose.yaw([x_hip, y_hip])*pi/180

            r.sleep()
    
    def states_callback(self,data):
        self.robot_position = data.pose[2].position   
    
    def shutdown(self):
        control.land()

def main():
    try:
        Yaw()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()