# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 15:57:29 2021

@author: Jake Thomas

This script generates an image tat is a normalised compilation of all letterforms superimposed.
"""

import itertools
import os
import matplotlib.pylab as plt
import numpy as np
import cv2

image_compilation = np.zeros((150,150))
counter = 0

letterform_directory = r'C:\Users\jake_\OneDrive\Pictures\Letterforms'
for letterform_names in os.listdir(letterform_directory):
    image_path = letterform_directory + os.sep + letterform_names
    image = cv2.imread(image_path, 0)
    plt.figure(counter)
    counter += 1
    for rows in range(0, 150):
        for columns in range(0, 150):
            image_compilation[rows][columns] = image_compilation[rows][columns] + image[rows][columns]
    plt.imshow(image_compilation, cmap = 'gray')