#! /usr/bin/env python

import cv2
import numpy as np
import time
import math


class Plot:
    def __init__(self, w=640, h=480, ros_rate=10):
        self.w = w
        self.h = h
        self.interval = 1/ros_rate
        self.imgPlot = np.zeros((self.h, self.w, 3), np.uint8)
        self.imgPlot[:] = 225, 225, 225

        cv2.rectangle(self.imgPlot, (0, 0),
                      (self.w, self.h),
                      (0, 0, 0), cv2.FILLED)
                      

    def update(self):
        self.drawBackground()

        return self.imgPlot

    def drawBackground(self):
        cv2.rectangle(self.imgPlot, (0, 0), (self.w, self.h), (0, 0, 0), cv2.FILLED)

        cv2.line(self.imgPlot, (0, self.h//2), (self.w, self.h//2), (150, 150, 150), 2)

        # Draw Grid Lines
        for x in range(0, self.w, 50):
            cv2.line(self.imgPlot, (x, 0), (x, self.h), (50, 50, 50), 1)

        for y in range(0, self.h, 50):
            cv2.line(self.imgPlot, (0, y), (self.w, y), (50, 50, 50), 1)