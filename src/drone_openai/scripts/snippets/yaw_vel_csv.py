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

import csv
import os
import rospkg
rospack = rospkg.RosPack()
log_dir = os.path.join(rospack.get_path("drone_openai"), "scripts/logs/")
Header = ['Yaw angle', 'Yaw speed']
with open(log_dir + 'yaw_logs.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerow(Header)

from helpers.cvlib import Detection
detection = Detection()

from helpers.control import Control
control = Control()
fpv = [320, 480]


class Yaw(object):
    def __init__(self):
        rospy.init_node('yaw_node', anonymous=True)
        self.rate = rospy.Rate(10)

        rospy.Subscriber("/drone/front_camera/image_raw",Image,self.cam_callback)
        self.bridge_object = CvBridge()
        self.frame = None
        self.frame_id = 0

        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move_msg = Twist()
        self.yaw_logs = []

        control.takeoff()
        rospy.on_shutdown(self.shutdown)

        while not rospy.is_shutdown():
            if self.frame is not None:
                if self.frame_id < 1000:
                    self.frame_id = self.frame_id + 1
                    print(self.frame_id)
                else:
                    print("Logs done")
                    log_header = ['Yaw angle', 'Yaw speed']
                    with open(log_dir + 'yaw_logs.csv', 'w') as f: 
                        write = csv.writer(f) 
                        write.writerow(log_header)
                        write.writerows(self.yaw_logs)

                frame = deepcopy(self.frame)
                
                centroids = detection.detect(frame)
                if len(centroids)==0:
                    # To-do: fill in gaps
                    yaw_speed = 0
                else:
                    cent = centroids[0]
                    yaw_angle = degrees(atan(float(fpv[0]-cent[0])/(fpv[1]-cent[1])))

                    # ROS rate at 10 HZ and Yaw speed at 90 deg/s
                    # ROS rate can be interrupted by internal processes (sensor processing)
                    # Yaw speed can be interrupted by both internal (battery) and external conditions (wind)
                    # System Identification
                    if yaw_angle > 9: yaw_speed = 90
                    elif yaw_angle < -9: yaw_speed = -90
                    elif yaw_angle < 9 and yaw_angle > 0: yaw_speed = yaw_angle * 10
                    elif yaw_angle < 0 and yaw_angle > -9: yaw_speed = yaw_angle * 10
                    else: yaw_speed = 0

                    self.yaw_logs.append([yaw_angle, yaw_speed])

                self.move_msg.angular.z = yaw_speed
                self.pub_cmd_vel.publish(self.move_msg)

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