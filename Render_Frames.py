# -*- coding: utf-8 -*-
"""
ECE4094 Test file: Jake Thomas

The purpose of this file is to develop the phosphene rendering algorithms that best capture information about the environment

"""
import itertools
import os
import time
import matplotlib.pylab as plt
import numpy as np
import cv2
import re
import random as rand

param = 10
phosphene_count = 10
frame_count = 0

#The next 3 functions allow for alphanumeric sorting that takes into account human sorting issues such as putting abc2 before abc10.

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

def empty(a):
  pass

def stackImages(scale,imgArray): 
    """ Stacks multiple images in the same window in an array-like fashion
    """    
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

"""
The next three functions are used to generate random and uniform gridpoints
"""

def cartesian_plot(x_param, y_param):
  x = [a for a in range(x_param)]
  y = [b for b in range(y_param)]

  return x,y

def rescale_cartesian(x_pixels, y_pixels, x_cartesian, y_cartesian):
  x_spacing = x_pixels/len(x_cartesian)
  y_spacing = y_pixels/len(y_cartesian)

  x_rescaled = []
  y_rescaled = []

  for x in range(len(x_cartesian)):
    x_rescaled = np.append(x_rescaled, x_cartesian[x]*x_pixels/len(x_cartesian) + x_spacing/2)
  for y in range(len(y_cartesian)):
    y_rescaled = np.append(y_rescaled, y_cartesian[y]*y_pixels/len(y_cartesian) + y_spacing/2)
  
  return x_rescaled,y_rescaled, x_spacing, y_spacing

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
    """
    Parameters
    ----------
    img : np.array 
        (n,m,1) np.array where n,m are image dims, greyscale images used.
    coords : np.array
        Set of gridpoints used for brightness mapping.

    Returns
    -------
    Brightness : np.array
        Array of same dimensions as coords, with appended neighbourhood brightness at each location.
    img_segment : np.array
        Image segment corresponding to the neighbourhood about the given gridpoint.

    """

    Brightness = np.empty([param, param])

    for i in range(np.size(coords, 0)):
        for j in range(np.size(coords, 1)):
            
            #Calculating neighbourhood about each point
            x_min = int(coords[i,j,0] - np.floor(x_spacing/2))
            y_min = int(coords[i,j,1] - np.floor(y_spacing/2))
            x_max = int(coords[i,j,0] + np.floor(x_spacing/2))
            y_max = int(coords[i,j,1] + np.floor(y_spacing/2))
            
            img_segment = img[y_min:y_max, x_min:x_max]
            Brightness[i,j] = sum(sum(img_segment))

    return Brightness, img_segment

def brightest_coords(Brightness_Array, phosphene_count, coords):
    """

    Parameters
    ----------
    Brightness_Array : np.array
        Array containing the gridpoints with the brightest neighbourhoods.
    phosphene_count : int
        parameter defining the allowable limit on concurrent phosphene excitations.
    coords : np.array
        Gridpoints.

    Returns
    -------
    active_phosphenes : np.array
        Array containing the coords of the phosphenes to be excited.

    """
    #Ranking the brightest neighbourhoods
    Ranked_Brightness = np.sort(Brightness_Array, axis = None)
    Brightest_n = Ranked_Brightness[(len(Ranked_Brightness) - phosphene_count):len(Ranked_Brightness)]
    mask = np.zeros_like(Brightness_Array)

    for b in Brightest_n:
        index = np.where(Brightness_Array == b, 1, 0)
        mask = np.logical_or(index, mask)

    #No phosphenes generated if insufficient detection occurs
    if sum(sum(mask)) > phosphene_count:
        mask = np.zeros_like(mask)
      
    active_phosphenes = coords[mask]

    return active_phosphenes

def Phosphene_map(phosphene_locations, phosphene):
    """

    Parameters
    ----------
    phosphene_locations : np.array
        Array containing locations of phosphenes to be activated.
    phosphene : np.array
        greyscale render of a single phosphene to be superimposed on relevant grid locations.

    Returns
    -------
    background : np.array
        Returns the np.array corresponding to a black background, with phosphenes superimposed in the phosphene_locations given.

    """
    
    background = np.zeros_like(Img)
  
    phosphene_x_pixels = len(phosphene[1,:])
    phosphene_y_pixels = len(phosphene[:,1])
    
    #Superimposing phosphenes in correct locations
    for locs in range(len(phosphene_locations)):
        p_loc = phosphene_locations[locs,:]
        
        x_min = int(np.ceil(p_loc[0] - phosphene_x_pixels/2))
        x_max = int(np.ceil(p_loc[0] + phosphene_x_pixels/2))
        y_min = int(np.ceil(p_loc[1] - phosphene_y_pixels/2))
        y_max = int(np.ceil(p_loc[1] + phosphene_y_pixels/2))
        
        background[max(y_min,0):min(y_max,y_pixels), max(x_min,0):min(x_max,x_pixels)] = background[max(y_min,0):min(y_max,y_pixels), max(x_min,0):min(x_max,x_pixels)] + phosphene

    return background

#Alternative to Phosphene_map() in which adverse phosphene sizes included
def resized_map(phosphene_locations, phosphene):
    """
      Parameters
    ----------
    phosphene_locations : np.array
        Array containing locations of phosphenes to be activated.
    phosphene : np.array
        greyscale render of a single phosphene to be superimposed on relevant grid locations.

    Returns
    -------
    background : np.array
        Returns the np.array corresponding to a black background, with phosphenes superimposed in the phosphene_locations given.

    """
    #Predefining phosphene size limits
    min_size = 15
    max_size = 300
    
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
        #determine phosphene size at a given radial distance
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
        
        #Error correcting for cases in which larger phosphenes overflow from background edges.
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

# Display map currently used
def plot_coords(coords, x_pixels, y_pixels):
    counter = 0
    dim = coords.shape[0]
    coord_list = np.ndarray((100,2))
    x_coords = np.ndarray((100,1))
    y_coords = np.ndarray((100,1))
    for i in range(dim):
        for j in range(dim):
            [x,y] = coords[i,j,:]
            coord_list[counter,:] = [x,y]
            counter = counter + 1
            x_coords[counter - 1] = x
            y_coords[counter - 1] = y
            
            
    return coord_list, x_coords, y_coords

#Predefining a single frame and phosphene so mapping parameters don't need to be constantly updated

initialising_frame_path = r'C:\Users\jake_\OneDrive\Documents\frames\Frame_1.jpg'
intialising_frame = cv2.imread(initialising_frame_path, 0)
x_pixels = len(intialising_frame[1,:])
y_pixels = len(intialising_frame[:,1])
scale_percent = 20 # percent of original size
width = int(intialising_frame.shape[1] * scale_percent / 100)
height = int(intialising_frame.shape[0] * scale_percent / 100)
f_dim = (width, height)
img_cropped = intialising_frame[int(y_pixels/2):y_pixels, 0:x_pixels]
Img = cv2.resize(img_cropped, f_dim, interpolation = cv2.INTER_AREA)
X_pixels = len(Img[1,:])
Y_pixels = len(Img[:,1])
      
#Generating gridpoints    
coords = np.zeros((param,param,2))    
for i in range(param):
    for j in range(param):
        [x, y, x_spacing, y_spacing] = rescale_random(X_pixels, Y_pixels, list(range(0,param)), list(range(0,param)))
        coords[i,j,:] = [x[0][0], y[0][0]]
      
phosphene_original = cv2.imread(r'C:\Users\jake_\OneDrive\Pictures\Phosphene_rendered.jpg', 0)
phosphene_x_pixels = len(phosphene_original[1,:])
phosphene_y_pixels = len(phosphene_original[:,1])
scale_percent = 100*min(x_spacing, y_spacing)/phosphene_x_pixels
width = int(phosphene_original.shape[0] * scale_percent / 100)
height = int(phosphene_original.shape[1] * scale_percent / 100)
p_dim = (width, height)
phosphene = cv2.resize(phosphene_original, p_dim, interpolation = cv2.INTER_AREA)

#Defining a list of frame names as they appear in the local machine
frames = []
for k in range(6455):
    file_name = 'Frame_{}.jpg'.format(k)
    frames = np.append(frames, file_name)

def coords_to_points(coords):
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
    
    return points

#iterating over all maps and walks to generate render frames in each mode

paths = {1,2,3,4,5,6,7}
maps = {1,2,3,4,5}
for random_map in maps:
    coords = np.zeros((param,param,2))
    for i in range(param):
        for j in range(param):
            [x, y, x_spacing, y_spacing] = rescale_random(X_pixels, Y_pixels, list(range(0,param)), list(range(0,param)))
            coords[i,j,:] = [x[0][0], y[0][0]]
    points = coords_to_points(coords)
    
    for path in paths:
        output_loc = r'C:\Users\jake_\OneDrive\Documents\White_path\Path_{}\Render_Frames_Adverse\map_{}'.format(path, random_map)
        frame_count = 0
    
        #Reading each frame and performing the relevant image manipulation for path detection
        for subdir, dirs, files in os.walk(r'C:\Users\jake_\OneDrive\Documents\White_path\Path_{}\Original_Walk_Frames'.format(path)):
            for filename in sort_nicely(files):
                filepath = subdir + os.sep + filename
                frame = cv2.imread(filepath, 0)
                img_original = cv2.imread(filepath)    
                frame_count = frame_count + 1
                
                #removing top half of image as paths are detected in lower FoV
                img_cropped = img_original[int(y_pixels/2):y_pixels, 0:x_pixels]
                img = cv2.resize(img_cropped, f_dim, interpolation = cv2.INTER_AREA)
                imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                
                #Filter values are hardcoded as they were validated in Test_Render_Params.py
                lower = np.array([0,0,60])
                upper = np.array([100,150,255])
                #Masks relevant pixels in original image
                mask = cv2.inRange(imgHSV,lower,upper)
                imgResult = cv2.bitwise_and(img,img,mask=mask)
                #Blurring image as required with Canny edge detection
                img_blur = cv2.GaussianBlur(imgResult, (63,63), 50)
                img_edge = cv2.Canny(img_blur, 31, 31)
                [Brightness, IMG] = brightness(img_edge, coords)
                active_phosphenes =  brightest_coords(Brightness, phosphene_count, coords)
                Map = resized_map(active_phosphenes, phosphene)
                
                m_scale_percent = 500 # percent of original size
                m_width = int(Map.shape[1] * m_scale_percent / 100)
                m_height = int(Map.shape[0] * m_scale_percent / 100)
                m_dim = (m_width, m_height)
                
                Map_resized = cv2.resize(Map, m_dim, interpolation = cv2.INTER_AREA)
                
                #Stacking images for presentation
                imgStack = stackImages(0.6,([img,img_edge],[mask,imgResult],[imgHSV,Map_resized]))
                cv2.imshow("Stacked Images", imgStack)
                
                #writting each render frame to the correct directory
                os.chdir(output_loc)
                file_name = 'White_path_frame_{}.jpg'.format(frame_count)
                cv2.imwrite(file_name, Map_resized)
                cv2.waitKey(1)

    #printing map coords to csv
    map_output_path = output_loc + os.sep + 'map_{}.csv'.format(random_map)
    np.savetxt(map_output_path, points, delimiter = ",")
    
cv2.destroyAllWindows()
