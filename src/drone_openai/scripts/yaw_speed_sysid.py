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
import time

from helpers.cvlib import Detection
detection = Detection()

from helpers.control import Control
control = Control()
fpv = [320, 480]


class Yaw(object):
    def __init__(self):
        rospy.init_node('yaw_node', anonymous=True)
        self.rate = rospy.Rate(1)
        self.current_yaw = 0.0

        rospy.Subscriber("/drone/front_camera/image_raw",Image,self.cam_callback)
        self.bridge_object = CvBridge()
        self.frame = None
        self.frame_id = 0

        rospy.Subscriber ('/drone/gt_pose', Pose, self.pose_callback)
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move_msg = Twist()

        control.takeoff()
        rospy.on_shutdown(self.shutdown)

        while not rospy.is_shutdown():
            if self.frame is not None:
                if self.frame_id == 10 or self.frame_id == 20:
                    print("Yaw 90 degree")
                    # Yaw speed: 90 deg/sec -> rospy.Rate 1HZ or time.sleep(1) to yaw 90 degree
                    # Using time sleep will interrupt ros frequency
                    self.move_msg.angular.z = 90
                    self.pub_cmd_vel.publish(self.move_msg)
                    time.sleep(1)
                else:
                    self.move_msg.angular.z = 0
                    self.pub_cmd_vel.publish(self.move_msg)
                    
                self.frame_id = self.frame_id + 1

            self.rate.sleep()
    
    def cam_callback(self,data):
        try:
            cv_img = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        self.frame = cv_img

    def pose_callback(self, data):
        orientation = data.orientation
        self.current_yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])[2]
    
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