#! /usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_srvs.srv import Empty

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


class Detect(object):
    def __init__(self):
        rospy.init_node('detect_node', anonymous=True)
        self.rate = rospy.Rate(10)
        self.current_yaw = 0.0

        rospy.Subscriber("/drone/front_camera/image_raw",Image,self.cam_callback)
        self.bridge_object = CvBridge()
        self.frame = None

        control.takeoff()
        rospy.on_shutdown(self.shutdown)

        while not rospy.is_shutdown():
            if self.frame is not None:
                start_time = time.time()
                frame = deepcopy(self.frame)
                current_yaw = deepcopy(self.current_yaw)
                
                centroids = detection.detect(frame)

                for cent in centroids:
                    cv2.circle(frame, (cent[0], cent[1]), 3, [0,0,255], -1, cv2.LINE_AA)
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

    def shutdown(self):
        control.land()

def main():
    try:
        Detect()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()