# -*- coding: utf-8 -*-
"""
Created on Sat May 22 13:30:21 2021

@author: jake_

This script was written to extract the frames from each walk and save them in a local directory
"""

import itertools
import os
import matplotlib.pylab as plt
import numpy as np
import cv2

def getFrames(input_loc, output_loc):
    """

    Parameters
    ----------
    input_loc : String
        Location of video file from which frames are to be extracted.
    output_loc : String
        Directory to which frames will be saved.

    Returns
    -------
    None.

    """
    count = 0
    frame_count = 0
    vidcap = cv2.VideoCapture(input_loc)
    success, image = vidcap.read()
    while success::
      count += 1
      success, frame = vidcap.read()
      frame_count += 1
      os.chdir(output_loc)
      file_name = 'Frame_{}.jpg'.format(frame_count)
      cv2.imwrite(file_name, frame)

#Paths to be iterated over
paths = {1, 2, 3, 4, 5, 6, 7}

#Generating frames from each path
for path in paths:
    input_loc = r'C:/Users/jake_/OneDrive/Documents/White_path/Path_{}/Original_Walk/Path_{}.mp4'.format(path, path)
    output_loc = r'C:/Users/jake_/OneDrive/Documents/White_path/Path_{}/Original_Walk_Frames'.format(path)
    img = getFrames(input_loc, output_loc)

