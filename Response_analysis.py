# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:47:33 2021

@author: Jake Thomas
This script is used to analyse the data from the mobility task
"""

import numpy as np
import matplotlib.pylab as plt
import os, os.path
import moviepy.editor as movie
import csv
from matplotlib.axis import Axis
import re

global duration

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

def GetResultData(Participant, Path, Map, Attempt, delay):
    """

    Parameters
    ----------
    Participant : String
        Participant name/id.
    Path : Int
        Path number.
    Map : String
        Path name. {'path_1',.....,'path_5', 'uniform_path'}.
    Attempt : Int
        Path number.
    delay : Float
        Value calculated to correct for machin delays when playing videos.

    Returns
    -------
    True_Path_Array : np.array
        np.array containing calibration timings for specific path being analysed.
    Participant_Attempt_Array : np.array
        np.array contaiing participant timings for the path/map/attempt being analysed.
    True_Path : Alternate representation of True_Path_Array.
    Participant_Attempt : Alternate representation of Participant_Attempt_Array

    """
    global duration
    Participant_Attempt = []
    vid_path = r'C:\Users\jake_\OneDrive\Documents\White_path\Path_{}\Original_Walk\Path_{}.mp4'.format(Path, Path)
    video = movie.VideoFileClip(vid_path)
    duration = video.duration
    True_Path = np.loadtxt(r'C:\Users\jake_\OneDrive\Documents\White_path\Path_{}\Original_Walk\Calibration_Path_{}.txt'.format(Path, Path))
    True_Path = True_Path
    True_Path_Array = True_Path.reshape(int(len(True_Path)/2), 2)
    Participant_Attempt_Path = r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\path_{}\{}\{}\user_response.csv'.format(Participant, Path, Map, Attempt)
    with open(Participant_Attempt_Path) as csv_file:
        Participant_Attempt_Array = [tuple(row) for row in csv.reader(csv_file)]
    for (x,y) in Participant_Attempt_Array:
        Participant_Attempt.append(float(x))
        Participant_Attempt.append(float(y))
    Participant_Attempt = Participant_Attempt - delay   
    
    return True_Path_Array, Participant_Attempt_Array, True_Path, Participant_Attempt


def delay_correction(user, path):
    """
    Parameters
    ----------
    user : String
        User name/id.
    path : Int
        Path number.

    Returns
    -------
    delay_correction : Float
        Value of average delay between initial calibration and participant response for a given path.

    """
    delay_vector = []
    initial_keystrokes = []
    map_path = r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\path_{}'.format(user, path)
    calibration_path = r'C:\Users\jake_\OneDrive\Documents\White_path\Path_{}\Original_Walk\Calibration_Path_{}.txt'.format(path, path)
    calibration_array = np.loadtxt(calibration_path)
    initial_calibration_keystroke = calibration_array[0]
    for maps in os.listdir(map_path):
        attempt_path = map_path + os.sep + maps
        for attempts in os.listdir(attempt_path):
            user_response_path = attempt_path + os.sep + attempts + os.sep + 'user_response.csv'
            with open(user_response_path) as csv_file:
                keystrokes = [x for x in csv.reader(csv_file)]
                initial_keystrokes = np.append(initial_keystrokes, float(keystrokes[0][0]))
    avg_initial_keystroke = np.mean(initial_keystrokes)
    delay_correction = avg_initial_keystroke - initial_calibration_keystroke
    
    return delay_correction
            
#Generates the array corresponding to key releases as opposed to presses.
def NonPathArray(True_Path):
    """
    Parameters
    ----------
    True_Path : np.array
        Array containing calibration timings.

    Returns
    -------
    Non_Path : np.array
        Returns the array that is functionally opposite to True_Path.

    """
    global duration
    Non_Path_Array = []
    if True_Path[0] == 0:
        Non_Path_Array.append[float(True_Path[1])]
        for item in True_Path[1:(len(True_Path))]:
            Non_Path_Array.append(float(item))
    else:
        Non_Path_Array.append(0)
        for item in True_Path[0:(len(True_Path))]:
            Non_Path_Array.append(float(item))
    if float(Non_Path_Array[-1]) > duration:
        Non_Path = Non_Path_Array[0:(len(Non_Path_Array)-1)]
    else:
        Non_Path = np.append(Non_Path_Array, duration)
        
    return Non_Path
        

#Need to score both ability to recognise path and ability to recognise non-paths

def Performance(True_Path_Array, Participant_Attempt_Array):
    """
    Parameters
    ----------
    True_Path_Array : np.array
        Calibration timings.
    Participant_Attempt_Array : np.array
        Participant timings.

    Returns
    -------
    Total_Correct : Float
        total time in agreement with calibration.
    Path_Time : Float
        Time in agreement with key presses (i.e. how correct participant was at identifying the path).
    Path_Score : Float
        Percentage score. weighted sum of participants ability to identify alignment and misalignment with path.

    """
    Total_Correct = 0
    Path_Time = 0
    for i in range(len(True_Path_Array)):
        [t_start, t_end] = True_Path_Array[i,:]
        Path_Time = Path_Time + (t_end - t_start)
        for j in range(len(Participant_Attempt_Array)):
            [p_start, p_end] = Participant_Attempt_Array[j][:]
            p_start = float(p_start)
            p_end = float(p_end)
            if (p_start < t_end) and (p_start > t_start):
                if (p_end < t_end):
                    correct_time = p_end - p_start
                    Total_Correct = Total_Correct + correct_time
                else:
                    correct_time = t_end - p_start
                    Total_Correct = Total_Correct + correct_time
            if (p_start < t_end) and (p_start < t_start):
                if (p_end < t_end) and (p_end > t_start):
                    correct_time = p_end - t_start
                    Total_Correct = Total_Correct + correct_time
    
    Path_Score = 100*Total_Correct/Path_Time
                        
    return Total_Correct, Path_Time, Path_Score

#Generate performances for each participant, each map and each map
def generate_performances():
    for users in os.listdir(r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses'):
        for paths in [1,2,3,4,5,6,7]:
            for maps in os.listdir(r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\path_{}'.format(users, paths)):
                for attempts in os.listdir(r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\path_{}\{}'.format(users, paths, maps)):
                    Delay = delay_correction(users, paths)
                    print('User', users,'Path', paths,'Map', maps,'Attempt', attempts)
                    Score = weighted_score(users, paths, maps, attempts, Delay)
                    print('Score: ', Score)
                    score_output_path = r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\path_{}\{}\{}\performance.txt'.format(users, paths, maps, attempts)
                    file = open(score_output_path, 'w') 
                    file.write(str(Score)) 
                    file.close() 

#Generates afformentioned weighted score
def weighted_score(Participant, Path, Map, Attempt, delay):
    [True_Path_Array, Participant_Attempt_Array, True_Path, Participant_Attempt] = GetResultData(Participant, Path, Map, Attempt, delay)
    True_Non_Path = NonPathArray(True_Path)
    True_Non_Path_Array = np.array(True_Non_Path)
    True_Non_Path_Array = True_Non_Path_Array.reshape(int(len(True_Non_Path_Array)/2), 2)
    Participant_Non_Path = NonPathArray(Participant_Attempt)
    Participant_Non_Path_Array = np.array(Participant_Non_Path)
    Participant_Non_Path_Array =        Participant_Non_Path_Array.reshape(int(len(Participant_Non_Path_Array)/2), 2)
    
    Path_Performance = Performance(True_Path_Array, Participant_Attempt_Array)
    Non_Path_Performance = Performance(True_Non_Path_Array, Participant_Non_Path_Array)
    Weighted_Score = Path_Performance[2]*(Path_Performance[1]/(Path_Performance[1] + Non_Path_Performance[1])) + Non_Path_Performance[2]*(Non_Path_Performance[1]/(Path_Performance[1] + Non_Path_Performance[1]))
    
    return Weighted_Score

generate_performances()

#Read score from file for given participant, path, map and attempt
def get_score(User, Path, Map, Attempt):
    score_path = r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\{}\{}\{}\performance.txt'.format(User, Path, Map, Attempt)
    score = np.loadtxt(score_path)
    return score

non_uniform_maps = ['map_1', 'map_2', 'map_3', 'map_4', 'map_5']

#The remaining code generates miscelaneous box plots contrasting map performance for each user

#Generating x_ticks for box plot

h_array = []
x_array = []

for random_maps in non_uniform_maps:
    metric_path = r'C:\Users\jake_\OneDrive\Documents\White_path\Maps\{}_metrics.csv'.format(random_maps)
    with open(metric_path) as csv_file:
        metrics = [tuple(row) for row in csv.reader(csv_file)]
        h = float(metrics[0][2])
        x = float(metrics[0][4])
        
    h_array = np.append(h_array, h)
    x_array = np.append(x_array, x)
        
h_array = np.sort(h_array)
x_array = np.sort(x_array)

h_x_ticks = []
x_x_ticks = []

for sorted_h in h_array:
    for random_maps in non_uniform_maps:
        metric_path = r'C:\Users\jake_\OneDrive\Documents\White_path\Maps\{}_metrics.csv'.format(random_maps)
        with open(metric_path) as csv_file:
            metrics = [tuple(row) for row in csv.reader(csv_file)]
            h = float(metrics[0][2])
        if h == sorted_h:
            h_x_ticks = np.append(h_x_ticks, random_maps)

h_x_ticks = [h_x_ticks[0]+ '\n h = 49.24',
             h_x_ticks[1]+ '\n h = 58.82',
             h_x_ticks[2]+ '\n h = 72.91',
             h_x_ticks[3]+ '\n h = 95.06',
             h_x_ticks[4]+ '\n h = 315.17']

print(h_x_ticks)
            
for sorted_x in x_array:
    for random_maps in non_uniform_maps:
        with open(metric_path) as csv_file:
            metrics = [tuple(row) for row in csv.reader(csv_file)]
            x = float(metrics[0][4])
        if x == sorted_x:
            x_x_ticks = np.append(x_x_ticks, random_maps)

x_x_ticks = [x_x_ticks[0]+ '\n x = 16.61',
             x_x_ticks[1]+ '\n x = 46.11',
             x_x_ticks[2]+ '\n x = 51.81',
             x_x_ticks[3]+ '\n x = 63.53',
             x_x_ticks[4]+ '\n x = 64.50']    

participant_number = 0
for users in os.listdir(r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses'):
    participant_number += 1
    
    map_1_scores = []
    map_2_scores = []
    map_3_scores = []
    map_4_scores = []
    map_5_scores = []
    uniform_map_scores = []
    
    for paths in os.listdir(r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}'.format(users)):
        path_maps = os.listdir(r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\{}'.format(users, paths))
        for maps in path_maps:
            for attempts in os.listdir(r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\{}\{}'.format(users, paths, maps)):
                score_path = r'C:\Users\jake_\OneDrive\Documents\White_path\User_Responses\{}\{}\{}\{}\performance.txt'.format(users, paths, maps, attempts)
                score = np.loadtxt(score_path)
                if (maps == non_uniform_maps[0]):
                    #append to map_1 results
                    map_1_scores = np.append(map_1_scores, score)
                if (maps == non_uniform_maps[1]):
                    #append to map_2 results
                    map_2_scores = np.append(map_2_scores, score)
                if (maps == non_uniform_maps[2]):
                    #append to map_3 results
                    map_3_scores = np.append(map_3_scores, score)
                if (maps == non_uniform_maps[3]):
                    #append to map_4 results
                    map_4_scores = np.append(map_4_scores, score)
                if (maps == non_uniform_maps[4]):
                    #append to map_5 results
                    map_5_scores = np.append(map_5_scores, score)
                if (maps == 'uniform_map'):
                    #append to uniform_map results
                    uniform_map_scores = np.append(uniform_map_scores, score)
    
    x_labels = np.append('Uniform \n h = 0', h_x_ticks)
    fig, ax = plt.subplots()
    ax.set_ylabel('Score')
    ax.set_ylim(0,100)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Different Maps')
    ax.set_title('Mobility task performance by map uniformity: Participant #{}'.format(participant_number))
    plt.boxplot([uniform_map_scores, map_2_scores, map_3_scores, map_1_scores, map_5_scores, map_4_scores])
    fig.tight_layout()
    plt.show()
    
    x_labels = np.append('Uniform \n x = 1.414', x_x_ticks)
    fig, ax = plt.subplots()
    ax.set_ylabel('Score')
    ax.set_ylim(0,100)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Different Maps')
    ax.set_title('Mobility task performance by map uniformity: Participant #{}'.format(participant_number))
    plt.boxplot([uniform_map_scores, map_1_scores, map_2_scores, map_3_scores, map_4_scores, map_5_scores])
    fig.tight_layout()
    plt.show()
    
    random_scores = [*map_1_scores, *map_2_scores, *map_3_scores, *map_4_scores, *map_5_scores]
    
    x_labels = ['Uniform', 'Random']
    fig, ax = plt.subplots()
    ax.set_ylabel('Score')
    ax.set_ylim(0,100)
    ax.set_xticklabels(x_labels)
    ax.set_title('Mobility task performance (Uniform Vs Random): Participant #{}'.format(participant_number))
    plt.boxplot([uniform_map_scores, random_scores])
    fig.tight_layout()
    plt.show()
    
            
        
