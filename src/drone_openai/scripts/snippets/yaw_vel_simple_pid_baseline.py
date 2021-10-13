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
hz = 10
fpv = [320, 480]

from simple_pid import PID
pid = PID(0.4,0.2,0.4, setpoint=0, sample_time=0.1)

class Yaw(object):
    def __init__(self):
        rospy.init_node('yaw_node', anonymous=True)
        self.rate = rospy.Rate(hz)

        rospy.Subscriber("/drone/front_camera/image_raw",Image,self.cam_callback)
        self.bridge_object = CvBridge()
        self.frame = None

        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move_msg = Twist()

        self.yaw_angle_pid = 0
        self.frame_id = 0
        self.yaw_logs = []

        control.takeoff()
        rospy.on_shutdown(self.shutdown)

        while not rospy.is_shutdown():
            if self.frame is not None:
                frame = deepcopy(self.frame)

                centroids = detection.detect(frame)
                if len(centroids)==0:
                    self.move_msg.angular.z = 0
                    self.pub_cmd_vel.publish(self.move_msg)
                else:
                    cent = centroids[0]
                    error = float(fpv[0]-cent[0])
                    self.error_pid = pid(error)

                    self.yaw_angle_pid = degrees(atan(float(self.error_pid/240)))

                    self.move_msg.angular.z = radians(self.error_pid) * hz
                    self.pub_cmd_vel.publish(self.move_msg)

                log_length = 250
                if self.frame_id < log_length:
                    self.yaw_logs.append(self.yaw_angle_pid)
                    
                if self.frame_id == log_length:
                    # Simple PID: 9.42 ~ 10.23 std
                    print("PID Baseline done")
                    yaw_logs_preprocessing = np.trim_zeros(np.array(self.yaw_logs))
                    print(self.yaw_logs)
                    std = statistics.stdev(yaw_logs_preprocessing)
                    print(std)

                self.frame_id = self.frame_id + 1

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