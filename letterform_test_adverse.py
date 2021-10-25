# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 18:56:44 2021

@author: Jake Thomas
This script runs the letterform recognition task using non-uniform maps with differing phosphene sizes
"""

import itertools
import os
import matplotlib.pylab as plt
import numpy as np
import cv2
import imageio
import re
import random as rand
from pynput.keyboard import Key, Listener
import string
from matplotlib.pyplot import imshow
import csv

global param
global phosphene_count

#Param = sqrt(grid resolution), posphene_count = # allowable concurrent phosphenes
param = 10
phosphene_count = 10

def initialise_parameters():
    initialising_frame_path = r'C:\Users\jake_\OneDrive\Pictures\Letterforms\A.png'
    intialising_frame = cv2.imread(initialising_frame_path, 0)
    x_pixels = len(intialising_frame[1,:])
    y_pixels = len(intialising_frame[:,1])

    coords = np.zeros((param,param,2))

    for i in range(param):
        for j in range(param):
            [x, y, x_spacing, y_spacing] = rescale_random(x_pixels, y_pixels, list(range(0,param)), list(range(0,param)))
            coords[i,j,:] = [x[0][0], y[0][0]]
            
    counter = 0
    dim = coords.shape[0]
    coord_list = np.ndarray((param**2,2))
    x_coords = np.ndarray((param**2,1))
    y_coords = np.ndarray((param**2,1))
    for i in range(dim):
        for j in range(dim):
            [x,y] = coords[i,j,:]
            coord_list[counter,:] = [x,y]
            counter = counter + 1
            x_coords[counter - 1] = x
            y_coords[counter - 1] = y
            
    new_coords = list(zip(x_coords, y_coords))
    X = np.transpose(x_coords)
    Y = np.transpose(y_coords)
    x = X[0,:]
    y = Y[0,:]
    points = list(zip(x,y))
    
    return x_pixels, y_pixels, coords, points, x_spacing, y_spacing

#Brightness mapping functions are identical to those found in rander frames. See Render_Frames.py for documentation on these functions

def rescale_random(x_pixels, y_pixels, x_random, y_random):
  x_spacing = x_pixels/len(x_random)
  y_spacing = y_pixels/len(y_random)
  x_min = 0 + int(np.floor(x_spacing/2))
  x_max = x_pixels - int(np.floor(x_spacing/2))
  y_min = 0 + int(np.floor(y_spacing/2))
  y_max = y_pixels - int(np.floor(y_spacing/2))
  x_rescaled = [rand.sample(range(x_min, x_max), 1)]
  y_rescaled = [rand.sample(range(y_min, y_max), 1)]

  return x_rescaled,y_rescaled, x_spacing, y_spacing

def brightness(img, coords):

  Brightness = np.empty([param, param])

  for i in range(np.size(coords, 0)):
    for j in range(np.size(coords, 1)):

      x_min = int(coords[i,j,0] - np.floor(x_spacing/2))
      y_min = int(coords[i,j,1] - np.floor(y_spacing/2))
      x_max = int(coords[i,j,0] + np.floor(x_spacing/2))
      y_max = int(coords[i,j,1] + np.floor(y_spacing/2))

      img_segment = img[y_min:y_max, x_min:x_max]
      Brightness[i,j] = sum(sum(img_segment))

  return Brightness

def brightest_coords(Brightness_Array, phosphene_count, coords):

  Ranked_Brightness = np.sort(Brightness_Array, axis = None)
  Brightest_n = Ranked_Brightness[(len(Ranked_Brightness) - phosphene_count):len(Ranked_Brightness)]
  mask = np.zeros_like(Brightness_Array)

  for b in Brightest_n:
    index = np.where(Brightness_Array == b, 1, 0)
    mask = np.logical_or(index, mask)
      
  active_phosphenes = coords[mask]

  return active_phosphenes

def Phosphene_map(phosphene_locations, phosphene):
    
    background = np.zeros_like(Img)
  
    phosphene_x_pixels = len(phosphene[1,:])
    phosphene_y_pixels = len(phosphene[:,1])

    for locs in range(len(phosphene_locations)):
        p_loc = phosphene_locations[locs,:]
        
        x_min = int(np.ceil(p_loc[0] - phosphene_x_pixels/2))
        x_max = int(np.ceil(p_loc[0] + phosphene_x_pixels/2))
        y_min = int(np.ceil(p_loc[1] - phosphene_y_pixels/2))
        y_max = int(np.ceil(p_loc[1] + phosphene_y_pixels/2))
        
        background[max(y_min,0):min(y_max,y_pixels), max(x_min,0):min(x_max,x_pixels)] = background[max(y_min,0):min(y_max,y_pixels), max(x_min,0):min(x_max,x_pixels)] + phosphene

    return background

def resized_map(phosphene_locations, phosphene):
    min_size = 20
    max_size = 150
    
    background = np.zeros_like(Img)
    phosphene_x_pixels = len(phosphene[1,:])
    phosphene_y_pixels = len(phosphene[:,1])
    x_pix = len(background[1,:])
    y_pix = len(background[:,1])
    
    k = 4*(max_size - min_size)/(x_pix**2 + y_pix**2)
    
    sizes = []
    
    for i in range(0, len(phosphene_locations)):
        x = phosphene_locations[i,0]
        y = phosphene_locations[i,1]
        x_dist = abs(x-x_pix/2)
        y_dist = abs(y-y_pix/2)
        size = int(np.floor(min_size + k * (x_dist**2 + y_dist**2)))
        p_dim = (size, size)
        phosphene = cv2.resize(phosphene, p_dim, interpolation = cv2.INTER_AREA)
        phosphene_x_pixels = len(phosphene[1,:])
        phosphene_y_pixels = len(phosphene[:,1])
        
        p_loc = phosphene_locations[i,:]
        
        x_min = int(np.ceil(p_loc[0] - phosphene_x_pixels/2))
        x_max = int(np.ceil(p_loc[0] + phosphene_x_pixels/2))
        y_min = int(np.ceil(p_loc[1] - phosphene_y_pixels/2))
        y_max = int(np.ceil(p_loc[1] + phosphene_y_pixels/2))
        
        if x_min < 0:
            x_min = int(0)
            x_max = int(phosphene_x_pixels)
        if y_min < 0:
            y_min = int(0)
            y_max = int(phosphene_y_pixels)
        if x_max > X_pixels:
            x_min = int(X_pixels - phosphene_x_pixels)
            x_max = int(X_pixels)
        if y_max > Y_pixels:
            y_min = int(Y_pixels - phosphene_y_pixels)
            y_max = int(Y_pixels)

        background[y_min:y_max, x_min:x_max] = background[y_min:y_max, x_min:x_max] + phosphene  
        
    return background

#Functions relating to keylogger

#Log key press and character of key
def on_press(key): # gets re-called a key repeats
     global break_program
     global user_input
     if key not in keys_pressed:  # if not repeat event
         Key = key.char
         user_input = np.append(user_input, Key.upper())
         break_program = True
         return False
             
#transition to next letterform upon key release                                                   
def on_release(key):
     del keys_pressed[key] # remove key from active list  
     if key==Key.esc:
         return False

#Generate and display list of letterform renders
def generate_random_letterform():
    global letterform
    global order
    global user_input
    letters = string.ascii_letters[26:52]
    letterforms = rand.sample(letters, 26)
    global x_spacing
    global y_spacing
    global x_pixels
    global y_pixels
    global X_pixels
    global Y_pixels
    global Img
    global img_original
    global keys_pressed
    
    user_input = []
    [x_pixels, y_pixels, coords, points, x_spacing, y_spacing] = initialise_parameters()        
    plt.plot(points[:][:], '*')
    plt.show()
    output_directory = r"C:\Users\jake_\Test_Project\Letter_Recognition\{}".format(Participant_Name)
    
    # Create target Directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    attempt_number = len([os.path.isdir(i) for i in os.listdir(output_directory)])
    print(attempt_number)
    attempt_directory = output_directory + os.sep + 'attempt_{}'.format(attempt_number)

    if not os.path.exists(attempt_directory):
        os.mkdir(attempt_directory)

    for letterform in letterforms:
        
        keys_pressed = {}       
        
        filepath = r'C:\Users\jake_\OneDrive\Pictures\Letterforms\{}.png'.format(letterform)
        frame = cv2.imread(filepath, 0)
        
        scale_percent = 100 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        f_dim = (width, height)
        img_cropped = frame[int(y_pixels/2):y_pixels, 0:x_pixels]
        Img = cv2.resize(img_cropped, f_dim, interpolation = cv2.INTER_AREA)
        X_pixels = len(Img[1,:])
        Y_pixels = len(Img[:,1])
    
        phosphene_original = cv2.imread(r'C:\Users\jake_\OneDrive\Pictures\Phosphene_rendered.jpg', 0)
        phosphene_x_pixels = len(phosphene_original[1,:])
        phosphene_y_pixels = len(phosphene_original[:,1])
        scale_percent = 100*min(x_spacing, y_spacing)/phosphene_x_pixels
        width = int(phosphene_original.shape[0] * scale_percent / 100)
        height = int(phosphene_original.shape[1] * scale_percent / 100)
        p_dim = (width, height)
        phosphene = cv2.resize(phosphene_original, p_dim, interpolation = cv2.INTER_AREA)
        
        img_original = cv2.imread(filepath)
        imgHSV = cv2.cvtColor(img_original,cv2.COLOR_BGR2HSV)
        lower = np.array([0,0,65])
        upper = np.array([179,90,255])
        mask = cv2.inRange(imgHSV,lower,upper)
        Brightness = brightness(mask, coords)
        active_phosphenes =  brightest_coords(Brightness, phosphene_count, coords)
        Map = resized_map(active_phosphenes, phosphene) 
        Brightness = brightness(mask, coords)
        active_phosphenes =  brightest_coords(Brightness, phosphene_count, coords)
        Map = resized_map(active_phosphenes, phosphene)
        imshow(Map, cmap='gray')
        plt.show()
            
        break_program = False
        #Listen to keyboard inputs after letterform is displayed
        with Listener(on_press=on_press,on_release=on_release) as listener:
            while break_program == False:
                listener.join()
                break
    
    #Names of .csv files to be produced
    map_filename = 'attempt_{}_map.csv'.format(attempt_number)
    order_filename = 'attempt_{}_order.csv'.format(attempt_number)
    user_input_filename = 'attempt_{}_user_input.csv'.format(attempt_number)
    
    map_output_path = attempt_directory + os.sep + map_filename
    order_output_path = attempt_directory + os.sep + order_filename
    user_input_output_path = attempt_directory + os.sep + user_input_filename
    
    letters = list(letterforms)
    inputs = list(user_input)
    
    np.savetxt(map_output_path, points, delimiter = ",")
    
    with open(order_output_path,'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(letters)
    
    with open(user_input_output_path,'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows(inputs)
    
    param_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}\Paramters.csv'.format(Participant_Name)
    params = [attempt_number, param**2, phosphene_count]
    
    with open(param_path, 'a', newline='') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(params)
    
    return letterforms

order = np.array([])
user_input = np.array([])

Participant_Name = input("Enter First Name:")
print("Participant: " + Participant_Name)

while(True):
    generate_random_letterform()