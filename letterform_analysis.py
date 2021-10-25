# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 02:31:22 2021

@author: Jake Thomas
This script
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
from scipy.spatial import Voronoi, voronoi_plot_2d
import string
from matplotlib.pyplot import imshow
import csv
from matplotlib.axis import Axis

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

def read_results(user, attempt):
    """
    Parameters
    ----------
    user : String
        User name/id.
    attempt : Int
        Attempt number.

    Returns
    -------
    user_input : List
        List containing user input for a given attempt.
    letterform_order : List
        List containing actual order of letterforms in a given attempt.
    points : np.array
        Grid points of map for specific user and attempt.

    """
    
    directory = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}\{}'.format(user, attempt)
    user_input_path = directory + os.sep + '{}_user_input.csv'.format(attempt)
    letterform_order_path = directory + os.sep + '{}_order.csv'.format(attempt)
    map_path = directory + os.sep + '{}_map.csv'.format(attempt)

    with open(user_input_path) as csv_file:
        user_input = [row for row in csv.reader(csv_file) if row]
    
    with open(letterform_order_path) as csv_file:
        letterform_order = [row for row in csv.reader(csv_file) if row]
        
    with open(map_path) as csv_file:
        points = [tuple(row) for row in csv.reader(csv_file)]        

    return user_input, letterform_order, points

#Calculates correct responses for a given attempt
def correct_responses(user_input, letterform_order):
    
    correct_user_responses = 0
    correct_indices = np.zeros((26,))
    
    for counter in range(len(user_input)):
        if (user_input[counter] == letterform_order[counter]):
            correct_user_responses += 1
            correct_indices[counter] = 1

    return correct_user_responses, correct_indices

#Generates the performances for each participant and attempt and prints the result to a .txt file
def generate_performances():
    user_path = r'C:\Users\jake_\Test_Project\Letter_Recognition'
    for users in os.listdir(user_path):
        attempt_list = []
        attempt_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}'.format(users)
        for attempts in os.listdir(attempt_path):
            attempt_value_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}\{}'.format(users, attempts)
            if os.path.isdir(attempt_value_path):
                if os.listdir(attempt_value_path):
                    attempt_list.append(attempts)        
        for nonempty_attempts in attempt_list:
            correct_array = []
            [user_input, letterform_order, points] = read_results(users, nonempty_attempts)
            [correct_user_responses, correct_indices] = correct_responses(user_input, letterform_order)
            correct_array = np.append(correct_array, correct_user_responses)
            output_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}\{}'.format(users, nonempty_attempts) + os.sep + '{}_score.txt'.format(nonempty_attempts)
            file = open(output_path, 'w') 
            file.write(str(100*sum(correct_array)/26)) 
            file.close() 
        
generate_performances()

#Generate a list containing the attmept numbers corresponding to a certain user, grid resolution, phosphene count and uniformity
def get_attempts(user, grid_resolution, phosphene_count, uniformity):
    #defining uniform map so that random and uniform attempts can be segregated
    
    uniform_map_1_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\Jake\attempt_93\attempt_93_map.csv'
    with open(uniform_map_1_path) as csv_file:
        uniform_points_1 = [tuple(row) for row in csv.reader(csv_file)]
        
    uniform_map_2_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\Jake\attempt_84\attempt_84_map.csv'
    with open(uniform_map_2_path) as csv_file:
        uniform_points_2 = [tuple(row) for row in csv.reader(csv_file)] 
    
    uniform_map_3_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\Jake\attempt_74\attempt_74_map.csv'
    with open(uniform_map_3_path) as csv_file:
        uniform_points_3 = [tuple(row) for row in csv.reader(csv_file)]
    
    attempt_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}'.format(user)
    attempt_list = []
    random_attempts = []
    uniform_attempts = []
    param_attempt_vector = []
    
    for attempts in os.listdir(attempt_path):
        attempt_value_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}\{}'.format(user, attempts)
        if os.path.isdir(attempt_value_path):
            if os.listdir(attempt_value_path):
                attempt_list.append(attempts)
    attempt_list = sort_nicely(attempt_list)
    
    #Selecting appropriate non-empty attempts from list:
    parameter_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}\Paramters.csv'.format(user)
    
    with open(parameter_path) as csv_file:
        parameters = [tuple(row) for row in csv.reader(csv_file)]
        parameters = parameters[1:len(parameters)][:]
        
    for (attempt_number, grid_size, p_count) in parameters:
        if ((int(grid_size) == grid_resolution) & (int(p_count) == phosphene_count)):
            param_attempt_vector = np.append(param_attempt_vector, attempt_number)
    param_attempt_list = ['attempt_{}'.format(a) for a in param_attempt_vector]
    
    for nonempty_attempts in attempt_list:
        map_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}\{}\{}_map.csv'.format(user, nonempty_attempts, nonempty_attempts)
        with open(map_path) as csv_file:
            points = [tuple(row) for row in csv.reader(csv_file)]
    
        if ((points == uniform_points_1) or (points == uniform_points_2) or (points == uniform_points_3)):
            uniform_attempts.append(nonempty_attempts)
        else:
            random_attempts.append(nonempty_attempts)
            
    if uniformity == 'random':
        intersection = list(set(param_attempt_list).intersection(random_attempts))
    if uniformity == 'uniform':
        intersection = list(set(param_attempt_list).intersection(uniform_attempts))
    
    return intersection       

#Generating box plots for each user by uniformity, as well as uniformity metrics

user_path = r'C:\Users\jake_\Test_Project\Letter_Recognition'
participant_number = 0
for Users in os.listdir(user_path):
    participant_number += 1
    
    avg_random_vector = []
    avg_uniform_vector = []
    
    avg_H1_vector = []
    avg_H2_vector = []
    avg_H3_vector = []
    
    avg_X1_vector = []
    avg_X2_vector = []
    avg_X3_vector = []
    
    for grid_sizes in [10**2, 15**2, 20**2]:
        for phosphene_counts in [10, 15, 20]:
            random_score_vector = []
            uniform_score_vector = []
            
            H1_score_vector = []
            H2_score_vector = []
            H3_score_vector = []
            
            X1_score_vector = []
            X2_score_vector = []
            X3_score_vector = []
            
            Random_attempts = get_attempts(Users, grid_sizes, phosphene_counts, 'random')
            Uniform_attempts = get_attempts(Users, grid_sizes, phosphene_counts, 'uniform')
            for specific_attempts in Random_attempts:
                metric_path = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}\{}\{}_map_metrics.csv'.format(Users, specific_attempts, specific_attempts)
                with open(metric_path) as csv_file:
                    metrics = [tuple(row) for row in csv.reader(csv_file)]
                    h = float(metrics[0][2])
                    x = float(metrics[0][4])
                avg_random_score_vector = []
                new_val = np.loadtxt(r'C:\Users\jake_\Test_Project\Letter_Recognition\{}\{}\{}_score.txt'.format(Users, specific_attempts, specific_attempts))
                random_score_vector = np.append(random_score_vector, new_val)
                #Binning results by metric values
                if h <= 25:
                    H1_score_vector = np.append(H1_score_vector, new_val)
                if ((h > 25) & (h <=50)):
                    H2_score_vector = np.append(H2_score_vector, new_val)
                if h > 50:
                    H3_score_vector = np.append(H3_score_vector, new_val)
                
                if x <= 25:
                    X1_score_vector = np.append(X1_score_vector, new_val)
                if ((x > 25) & (x <=50)):
                    X2_score_vector = np.append(X2_score_vector, new_val)
                if x > 50:
                    X3_score_vector = np.append(X3_score_vector, new_val)
                
            if len(random_score_vector):
                avg_random_score_vector = np.mean(random_score_vector)
                avg_random_vector = np.append(avg_random_vector, avg_random_score_vector)
                
            if len(H1_score_vector):
                avg_H1_score_vector = np.mean(H1_score_vector)
                avg_H1_vector = np.append(avg_H1_vector, avg_H1_score_vector)
            if len(H2_score_vector):
                avg_H2_score_vector = np.mean(H2_score_vector)
                avg_H2_vector = np.append(avg_H2_vector, avg_H2_score_vector)
            if len(H3_score_vector):
                avg_H3_score_vector = np.mean(H3_score_vector)
                avg_H3_vector = np.append(avg_H3_vector, avg_H3_score_vector)
                
            if len(X1_score_vector):
                avg_X1_score_vector = np.mean(X1_score_vector)
                avg_X1_vector = np.append(avg_X1_vector, avg_X1_score_vector)
            if len(X2_score_vector):
                avg_X2_score_vector = np.mean(X2_score_vector)
                avg_X2_vector = np.append(avg_X2_vector, avg_X2_score_vector)
            if len(X3_score_vector):
                avg_X3_score_vector = np.mean(X3_score_vector)
                avg_X3_vector = np.append(avg_X3_vector, avg_X3_score_vector)
                
            for specific_attempts in Uniform_attempts:
                avg_random_score_vector = []
                uniform_score_vector = np.append(uniform_score_vector, np.loadtxt(r'C:\Users\jake_\Test_Project\Letter_Recognition\{}\{}\{}_score.txt'.format(Users, specific_attempts, specific_attempts)))
            if len(uniform_score_vector):
                avg_uniform_score_vector = np.mean(uniform_score_vector)
                avg_uniform_vector = np.append(avg_uniform_vector, avg_uniform_score_vector)
                
            random_x = np.ones_like(random_score_vector)
            uniform_x = 2*np.ones_like(uniform_score_vector)
            data = np.array([random_score_vector, uniform_score_vector])
            
            x_labels = ['Random', 'Uniform']
            fig, ax = plt.subplots()
            plt.plot(random_x, random_score_vector, '*', uniform_x, uniform_score_vector, '*')
            ax.set_ylabel('Score')
            ax.set_ylim(0,100)
            ax.xaxis.set_ticks([0,1,2,3])
            ax.set_xticklabels(['','Random', 'Uniform',''])
            ax.set_xlabel('Render Conditions')
            ax.set_title('Letterform recognition scores: User = {}, Grid size = {}, Phosphene count = {}'.format(Users, grid_sizes, phosphene_counts))
            fig.tight_layout()
            plt.show()
    
    #h_metric plotting
    x_labels = ['Uniform (h = 0)', '0 > h > 25', '25 < h < 50', '50 < h']
    fig, ax = plt.subplots()
    ax.set_ylabel('Score')
    ax.set_ylim(0,100)
    ax.set_xticklabels(['Uniform (h = 0)', '0 > h > 25', '25 < h < 50', '50 < h'])
    ax.set_xlabel('grid uniformity metric')
    ax.set_title('Letterform recognition scores by grid uniformity: Participant #{}'.format(participant_number))
    plt.boxplot([avg_uniform_vector, avg_H1_vector, avg_H2_vector, avg_H3_vector])
    fig.tight_layout()
    plt.show()
    
    #x_metric plotting
    x_labels = ['Uniform (x = 1.414)', '0 > x > 25', '25 < x < 50', '50 < x']
    fig, ax = plt.subplots()
    ax.set_ylabel('Score')
    ax.set_ylim(0,100)
    ax.set_xticklabels(['Uniform (x = 1.414)', '0 > x > 25', '25 < x < 50', '50 < x'])
    ax.set_xlabel('grid uniformity metric')
    ax.set_title('Letterform recognition scores by grid uniformity: Participant #{}'.format(participant_number))
    plt.boxplot([avg_uniform_vector, avg_X1_vector, avg_X2_vector, avg_X3_vector])
    fig.tight_layout()
    plt.show()
    
    #Plotting random vs uniform results
    x_labels = ['Random', 'Uniform']
    fig, ax = plt.subplots()
    ax.set_ylabel('Score')
    ax.set_ylim(0,100)
    ax.set_xticklabels(['Random', 'Uniform'])
    ax.set_xlabel('Render Conditions')
    ax.set_title('Letterform recognition scores: Participant #{}'.format(participant_number))
    plt.boxplot([avg_random_vector, avg_uniform_vector])
    fig.tight_layout()
    plt.show()           

#Generate user statistics
def user_statistics(user):
    correct_array = []
    user_directory = r'C:\Users\jake_\Test_Project\Letter_Recognition\{}'.format(user)
    total_attempts = len([os.path.isdir(i) for i in os.listdir(user_directory)])
    attempt_vector = [int(i) for i in range(total_attempts)]
    for attempt in attempt_vector:
        [user_input, letterform_order, points] = read_results(user, attempt)
        [correct_user_responses, correct_indices] = correct_responses(user_input, letterform_order)
        correct_array = np.append(correct_array, correct_user_responses)
    return correct_array, attempt_vector