#! /usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
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
import statistics
import time

from helpers.cvlib import Detection
detection = Detection()

from helpers.control import Control
control = Control()
fpv = [320, 480]


class Yaw(object):
    def __init__(self):
        rospy.init_node('yaw_node', anonymous=True)
        self.hz = 10
        self.rate = rospy.Rate(self.hz)

        rospy.Subscriber("/drone/front_camera/image_raw",Image,self.cam_callback)
        self.bridge_object = CvBridge()
        self.frame = None
        
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move_msg = Twist()
        self.yaw_angle = 0

        self.frame_id = 0
        self.kp = 1
        self.yaw_logs = []

        control.takeoff()
        rospy.on_shutdown(self.shutdown)

        while not rospy.is_shutdown():
            if self.frame is not None:

                frame = deepcopy(self.frame)
                centroids = detection.detect(frame)

                if len(centroids)==0:
                    # To-do: fill in gaps
                    yaw_speed = 0
                else:
                    cent = centroids[0]
                    self.yaw_angle = degrees(atan(float(fpv[0]-cent[0])/(fpv[1]-cent[1])))

                    # ROS rate at 10 HZ and Yaw speed at 90 deg/s
                    # ROS rate can be interrupted by internal processes (sensor processing)
                    # Yaw speed can be interrupted by both internal (battery) and external conditions (wind)
                    # System Identification
                    if self.yaw_angle > 9: yaw_speed = 90
                    elif self.yaw_angle < -9: yaw_speed = -90
                    elif self.yaw_angle < 9 and self.yaw_angle > 4.5: yaw_speed = float(self.kp/10) * self.yaw_angle * self.hz
                    elif self.yaw_angle < -4.5 and self.yaw_angle > -9: yaw_speed = float(self.kp/10) * self.yaw_angle * self.hz
                    else: yaw_speed = 0

                self.move_msg.angular.z = yaw_speed
                self.pub_cmd_vel.publish(self.move_msg)

                log_length = 250
                if self.frame_id < log_length:
                    self.frame_id = self.frame_id + 1
                    self.yaw_logs.append(self.yaw_angle)
                elif self.kp < 10:
                    # Fixed Speed No Tuning: 12 std
                    # Ajusted Speed No Tuning: 14 std
                    # 0.6 kp within 45 ~ 90 deg/s
                    print("KP " + str(self.kp) + " Baseline done")
                    # print(self.yaw_logs)
                    std = statistics.stdev(self.yaw_logs)
                    print(std)

                    self.frame_id = 0
                    self.kp = self.kp + 1
                    self.yaw_logs = []

            self.rate.sleep()
    
    def cam_callback(self,data):
        try:
            cv_img = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        self.frame = cv_img
    
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