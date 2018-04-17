#!/usr/bin/env python

import sys
import os, sys, cv2
import numpy as np
import math as m

def E2R(Ry, Rx, Rz):
    R_yaw = np.array([[ m.cos(Ry), 0 ,m.sin(Ry)],
                      [ 0,         1 ,     0],
                      [-m.sin(Ry), 0 ,m.cos(Ry)]])
    R_pitch = np.array([[1, 0, 0],
                        [0, m.cos(Rx), -m.sin(Rx)],
                        [0, m.sin(Rx), m.cos(Rx)]])
    #R_roll = np.array([[[m.cos(Rz), -m.sin(Rz), 0],
    #                    [m.sin(Rz), m.cos(Rz), 0],
    #                    [ 0,         0 ,     1]])
    return (R_pitch.dot(R_yaw))

def Space2Image(P0, R0, pts3):
    pts2_norm = P0.dot(R0.dot(pts3))
    pts2 = np.array([int(pts2_norm[0]/pts2_norm[2]), int(pts2_norm[1]/pts2_norm[2])])
    return pts2

def image_3d(im_raw, calib, predictions, threshold)
