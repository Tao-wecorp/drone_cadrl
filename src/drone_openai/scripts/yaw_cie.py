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
        self.runtime = 0.0
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
        i = 0
        while not rospy.is_shutdown():
            if self.frame is not None:          
                i = i+1
                start_time = time.time()
                frame = deepcopy(self.frame)
                centroids = detection.detect(frame)
                if len(centroids)==0: 
                    print(str(i%10) + ": none")
                    continue
                else:
                    cent = centroids[0]
                    self.yaw_angle = control.yaw(cent)
                    print(str(i%10) + ": " + str(cent[0]) + ", " + str(cent[1]) + ", " + str(self.yaw_angle))
                rospy.sleep(time.time() - start_time)

            detect_rate.sleep()

    def yaw_cie(self):
        cie_rate = rospy.Rate(30)
        i = 0
        state_robot_msg = ModelState()
        state_robot_msg.model_name = 'sjtu_drone'
        while not rospy.is_shutdown():
            i = i+1
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                pose.position = self.robot_position
                print("CIE " + str(i%30) + ": " + str(self.yaw_angle))
                pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, self.yaw_angle*pi/180))                  
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