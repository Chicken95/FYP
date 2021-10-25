# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:13:00 2021

@author: Jake Thomas
This script collates all of the render frames from a given walk into a video to be played in the psychophysics experiment
"""
import cv2
import numpy as np
import glob
import os
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number      chunks.
        "z23a" -> ["z", 23, "a"]
        """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    return sorted(l, key=alphanum_key)


paths = {1,2,3,4,5,6,7}
maps = {1,2,3,4,5}
for map_coords in maps:
    for path in paths:
        img_array = []
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    
        out_path = r'C:\Users\jake_\OneDrive\Documents\White_path\Path_{}\Render_Video_Adverse\map_{}\render_video.mp4'.format(path, map_coords) 
        for subdir, dirs, files in os.walk(r'C:\Users\jake_\OneDrive\Documents\White_path\Path_{}\Render_Frames_Adverse\map_{}'.format(path, map_coords)):
            for filename in sort_nicely(files):
                filepath = subdir + os.sep + filename
                img = cv2.imread(filepath)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)
            
            #Concatenating images and writing the final concatenation to a .mp4 file
            writer = cv2.VideoWriter(out_path, fourcc, 30, size)   
            for i in range(len(img_array)):
                writer.write(img_array[i])
            writer.release()
        #Clearing variables for memory
        del img_array, writer


