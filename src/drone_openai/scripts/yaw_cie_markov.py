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
from threading import Thread

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
        self.yaw_angle = 0.0
        self.yaw_angle_pre = 0.0
        self.yaw_angle_step = 0.0
        self.yaw_angle_cie = 0.0
        self.yaw_angle_cie_pre = 0.0

        self.frame = None
        self.bridge_object = CvBridge()
        
        # self.yaw_cie = 0.0
        self.robot_position = None

        # Sub & Pub
        rospy.Subscriber("/drone/front_camera/image_raw",Image,self.cam_callback)
        self.states_sub = rospy.Subscriber("/gazebo/model_states",ModelStates,self.states_callback)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Takeoff & Land
        control.takeoff()

        # Detect yaw
        self.detect_thr = Thread(target=self.yaw_detect, args=())
        self.detect_thr.daemon = True
        self.detect_thr.start()
        self.yaw_cie()

        rospy.on_shutdown(self.shutdown)
        rospy.spin()

    # Methods
    def cam_callback(self,data):
        try:
            cv_img = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        self.frame = cv_img
    
    def states_callback(self,data):
        self.robot_position = data.pose[-1].position
    
    def yaw_detect(self):
        detect_rate = rospy.Rate(10)
        n_none = 0
        while not rospy.is_shutdown():
            start_time = time.time()
            if self.frame is not None:
                frame = deepcopy(self.frame)
                centroids = detection.detect(frame)
                if len(centroids)==0: # decision gap
                    n_none = n_none + 1
                    self.yaw_angle = self.yaw_angle + self.yaw_angle_step
                    # if (self.yaw_angle > 0 and self.yaw_angle_step > 0) or (self.yaw_angle < 0 and self.yaw_angle_step < 0): 
                    #     self.yaw_angle = self.yaw_angle + self.yaw_angle_step * 1.1
                    # elif (self.yaw_angle > 0 and self.yaw_angle_step < 0) or (self.yaw_angle < 0 and self.yaw_angle_step > 0):
                    #     self.yaw_angle = self.yaw_angle + self.yaw_angle_step * 0.9
                    continue
                else: # decison made
                    cent = centroids[0]

                    self.yaw_angle = control.yaw(cent) # perception team's globel yaw

                    if n_none != 0:
                        self.yaw_angle_step = (self.yaw_angle - self.yaw_angle_pre) / (n_none + 1)
                        # avoid overshooting
                        self.yaw_angle_step = min(self.yaw_angle_step, 0.6)
                        self.yaw_angle_step = max(self.yaw_angle_step, -0.6)
                    self.yaw_angle_pre = self.yaw_angle
                    n_none = 0

                    cv2.circle(frame, (320, cent[1]), 3, [0,0,255], -1, cv2.LINE_AA)
                    cv2.circle(frame, (cent[0], cent[1]), 3, [0,255,0], -1, cv2.LINE_AA)

                cv2.imshow("", frame)
                cv2.waitKey(1)
                rospy.sleep(0.05)
                
            detect_rate.sleep()
            # print(time.time() - start_time)

    def yaw_cie(self):
        cie_rate = rospy.Rate(30)
        state_robot_msg = ModelState()
        state_robot_msg.model_name = 'sjtu_drone'
        start_time = time.time()
        while not rospy.is_shutdown():
            
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                pose.position = self.robot_position

                # no filters or gaps
                self.yaw_angle_cie = self.yaw_angle
                
                # markov smoothing or filters
                self.yaw_angle_cie = (self.yaw_angle + self.yaw_angle_cie_pre) / 2
                self.yaw_angle_cie_pre = self.yaw_angle_cie
                print("FPS30: " + str(self.yaw_angle_cie))

                pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, self.yaw_angle_cie*pi/180))                  
                state_robot_msg.pose = pose

                self.set_state(state_robot_msg)
            except rospy.ServiceException:
                pass
            
            cie_rate.sleep()
            

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