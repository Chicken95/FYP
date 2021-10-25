# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 20:21:38 2021

@author: Jake Thomas
This script runs the Mobility test under adverse conditions
"""

#Although this code is now used to run the test, it was originally used to generate calibration files by playing the original walks instead of the renders

from pynput.keyboard import Key, Listener
import time
from os import startfile
import numpy as np
import moviepy.editor as movie
import matplotlib.pylab as plt
import os, os.path
import random

Participant_Name = input("Enter First Name:")
print("Participant: " + Participant_Name)

#Keylogging
def on_press(key): # gets re-called a key repeats
     global path_initiated
     global path_deterction
     global counter
     global break_program
     if key not in keys_pressed:  # if not repeat event
         keys_pressed[key] = time.time() # add key and start time
         path_initiated = time.time() - start_time
         if (time.time() - start_time) > duration:
             #video has ended
             complete_detection = path_detection[0:counter,:]
             break_program = True
             output_loc = r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\path_{}\map_{}\attempt_{}\user_response.csv'.format(Participant_Name, Path_num, Map_num, attempt_number)
             np.savetxt(output_loc, complete_detection, delimiter = ",")
             return False
             
                                                   
def on_release(key):                  
     global counter                                                          
     global path_concluded
     global path_detection                 
     path_concluded = time.time() - start_time
     path_detection[counter, :] = [path_initiated, path_concluded]
     counter = counter + 1  
     del keys_pressed[key] # remove key from active list  
     if key==Key.esc:
         return False

def RunTest(pair, Participant_Name):
    global counter
    global start_time
    global keys_pressed  # dictionary of keys currently pressed
    global path_detection
    global path_initiated                                
    global path_concluded
    global duration
    global dirName
    global order_name
    global response_name
    global Path_num
    global Map_num
    global attempt_number
    counter = 0
    start_time = time.time()
    keys_pressed = {}
    path_detection = np.zeros((100,2))
    Path_num = pair[0][1]
    Map_num = pair[0][0]
       
    #Building the directories if necessary
    
    dirName = r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}'.format(Participant_Name)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    dirName = r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\path_{}'.format(Participant_Name, Path_num)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    dirName = r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\path_{}\map_{}'.format(Participant_Name, Path_num, Map_num)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    attempt_number = len([os.path.isdir(i) for i in os.listdir(dirName)])
    dirName = dirName + os.sep + 'attempt_{}'.format(attempt_number)
    # Create target Directory if it doesn't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        
    order_name = 'render_order.csv'
    response_name = 'user_response.csv'
    
    path = r'C:\Users\jake_\OneDrive\Documents\White_path\Path_{}\Render_Video_adverse\map_{}\render_video.mp4'.format(Path_num, Map_num)
    video = movie.VideoFileClip(path)
    duration = video.duration                 
    startfile(path)
    
    break_program = False
    with Listener(on_press=on_press,on_release=on_release) as listener:
        while break_program == False:
            listener.join()
            break

#Generate random sequence of 7 walks rendered using 5 random maps
map_order = [random.sample(range(1, 6), 5)]
walk_order = [random.sample(range(1, 8), 5)]

order_pairs = [tuple(pair) for pair in np.transpose([map_order, walk_order])]

for pair in order_pairs:
    RunTest(pair, Participant_Name)