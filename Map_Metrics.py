# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 18:58:41 2021

@author: jake_
"""
import itertools
import os
import math
import time
import matplotlib.pylab as plt
import numpy as np
import cv2
import imageio
import re
import random as rand
from scipy.spatial import Voronoi, voronoi_plot_2d
import string
from matplotlib.pyplot import imshow
import csv

#Generate area of convex polygon by shoestring method
def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

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

def map_covariance(x_points, y_points):
    """
    Parameters
    ----------
    x_points : np.array
        x coords of map.
    y_points : np.array
        y coords of map.

    Returns
    -------
    gamma : np.array
        Array containing minimum distances from each point to another.
    Cov : Float
        Covariance of gamma array.
    Lambda : Float
        Covaraince measure of point uniformity.

    """
    current = 0
    new = 0
    grid_resolution = len(x_points)
    gamma = np.ndarray((grid_resolution))
    for i in range(len(x_points)):
        x_current = x_points[i]
        y_current = y_points[i]
        Dist_Array = np.ndarray((grid_resolution - 1))
        Dist_array = np.zeros_like(Dist_Array)
        Dist_array = []
        for j in range(len(x_points)):
            if j != i:
                x_new = x_points[j]
                y_new = y_points[j]
                new = new + 1
                dist = (x_current-x_new)**2 + (y_current-y_new)**2
                Dist_array = np.append(Dist_array, dist)
        gamma[i] = np.sqrt(np.min(Dist_array[np.nonzero(Dist_array)]))
    Cov = np.cov(gamma)
    Lambda = 1/(np.mean(gamma))*Cov
        
    return gamma, Cov, Lambda
        
def point_distribution_norm(points, vor_lower, vor_upper):
    """
    Parameters
    ----------
    points : np.array
        map coords.
    vor_lower : Int
        lower bound on Voronoi tesselation.
    vor_upper : Int
        Upper bound on Voronoi tesselation.

    Returns
    -------
    h : Float
        Point distribution norm.
    H_array : np.array
        Array containing the maximum distances from a given point to the vertices of the Voronoi region containing it.
    mu : Float
        Point distribution ratio.
    v : Float
        Cell volume deviation.

    """
    vor = Voronoi(points) # Creates a Voroni diagram of the phosphene map
    vertices = vor.vertices # Coordinates of Voroni vertices
    regions = vor.regions # Indices of vertices forming each region
    point_region = vor.point_region # Index of region corresponding to each input coord
    
    # h = max{h_i} where h_i = max|z_i - y| i.e. h = max dis between a veronoi vertex and associated input point
    
    #First step is to calculate distances for each vertex to the points it's corresponding regions enclose
    
    index = 0
    H_array = np.zeros_like(point_region, dtype = float)
    Area_array = []
    for i in point_region:
        vertex_Indices = np.array(regions[i])
        vertex_indices = np.delete(vertex_Indices, np.argwhere(vertex_Indices == -1))
        vertex_points = vertices[vertex_indices]
        point = points[index]
        constrained_points = []
        for v_points in vertex_points:
            dist = []
            if (v_points[0]>vor_lower and v_points[1]>vor_lower and v_points[0]<vor_upper and v_points[1]<vor_upper):
                d = np.sqrt((point[0]-v_points[0])**2 + (point[1]-v_points[1])**2)
                constrained_points.append(tuple(v_points))
                dist = np.append(dist, d)
                max_dist = np.max(dist)
        x_coords = [x[0] for x in constrained_points]
        y_coords = [y[1] for y in constrained_points]
        Area = PolyArea(x_coords, y_coords)
        Area_array.append(Area)
        H_array[index] = float(max_dist)
        index = index + 1
    mask = np.where(np.array(Area_array) < 0.1)
    Area_array_new = np.delete(Area_array, mask)
    v = np.max(Area_array_new)/np.min(Area_array_new)
    h = np.max(H_array)
    mu = np.max(H_array)/np.min(H_array)
    
    return h, H_array, mu, v

#Read grid points from csv file
def get_points(map_path):
    
    with open(map_path) as csv_file:
        points = [tuple(row) for row in csv.reader(csv_file, quoting = csv.QUOTE_NONNUMERIC) if row]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    return points, x_coords, y_coords

#Generate uniformity metrics for each map
def generate_metrics(map_path, vor_lower, vor_upper):
    [points, x_coords, y_coords] = get_points(map_path)
    [gamma, Cov, Lambda] = map_covariance(x_coords, y_coords)
    mesh_ratio = np.max(gamma)/np.min(gamma)
    vor = Voronoi(points)
    [h, H_array, mu, v] = point_distribution_norm(points, vor_lower, vor_upper)
    chi = np.max(np.divide(2*H_array, gamma))
    metric_vector = [Lambda, mesh_ratio, h, mu, chi, v]
    return metric_vector

#Code for generating map uniformity metrics

#Generating map metrics for mobility task
for maps in [1,2,3,4,5]:
    path = r'C:\Users\jake_\OneDrive\Documents\White_path\Maps\map_{}.csv'.format(maps)
    output_path = r'C:\Users\jake_\OneDrive\Documents\White_path\Maps\map_{}_metrics.csv'.format(maps)
    metric_vector = generate_metrics(path, 0, 600)
    with open(output_path,'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerow(metric_vector)

#Generating map metrics for letterform task
for users in os.listdir(r'C:\Users\jake_\Test_Project\Letter_Recognition'):
    user_path = r'C:\Users\jake_\Test_Project\Letter_Recognition' + os.sep + '{}'.format(users)
    attempt_list = os.listdir(r'C:\Users\jake_\Test_Project\Letter_Recognition' + os.sep + '{}'.format(users))
    attempt_list = sort_nicely(attempt_list)
    attempt_list = attempt_list[1:-1]
    nonempty_attempts = []
    for attempt_number in attempt_list:
        attempt_path = user_path + os.sep + attempt_number
        print('attempt_path\n', attempt_path)
        if os.listdir(attempt_path):
            nonempty_attempts.append(attempt_number)
            output_path = attempt_path + os.sep + '{}_map_metrics.csv'.format(attempt_number)
            print(attempt_number)
            metric_vector = generate_metrics(attempt_path + os.sep + '{}_map.csv'.format(attempt_number), 0, 200)
            print(metric_vector)
            with open(output_path,'w') as result_file:
                wr = csv.writer(result_file, dialect='excel')
                wr.writerow(metric_vector)
    