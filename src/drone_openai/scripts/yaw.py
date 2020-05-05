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
kp = 0.5

class Yaw(object):
    def __init__(self):
        rospy.init_node('yaw_node', anonymous=True)
        self.rate = rospy.Rate(5)
        self.current_yaw = 0.0

        rospy.Subscriber("/drone/front_camera/image_raw",Image,self.cam_callback)
        self.bridge_object = CvBridge()
        self.frame = None

        rospy.Subscriber ('/drone/gt_pose', Pose, self.pose_callback)
        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.move_msg = Twist()

        control.takeoff()
        rospy.on_shutdown(self.shutdown)

        while not rospy.is_shutdown():
            if self.frame is not None:
                start_time = time.time()
                frame = deepcopy(self.frame)
                current_yaw = deepcopy(self.current_yaw)
                
                centroids = detection.detect(frame)
                if len(centroids)==0: 
                    continue
                else:
                    cent = centroids[0]
                    yaw_angle = degrees(atan(float(fpv[0]-cent[0])/(fpv[1]-cent[1])))
                    yaw_angular_z = yaw_angle*pi/180 - current_yaw

                    self.move_msg.angular.z = kp * yaw_angular_z
                    self.pub_cmd_vel.publish(self.move_msg)

                    cv2.circle(frame, (320, cent[1]), 3, [0,0,255], -1, cv2.LINE_AA)
                    cv2.circle(frame, (cent[0], cent[1]), 3, [0,255,0], -1, cv2.LINE_AA)

                cv2.imshow("", frame)
                cv2.waitKey(1)
                 
                print("%s seconds" % (time.time() - start_time))

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
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()