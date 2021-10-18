#!/usr/bin/python3
import cv2 as cv
import random
import numpy
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


im = cv.imread("/home/river-charles/kinova/src/artbot/src/test/output_images/k_means_layered/k_means_colors1.jpg")

bool_arr = []
for row in im:
    temp = []
    for pixel in row:
        temp.append(pixel[0] == 255 or pixel[1] == 255 or pixel[2] == 255)
    bool_arr.append(temp)

for i in range(len(bool_arr)):
    for j in range(len(bool_arr[i])):
        if(bool_arr[i][j]):
            im[i][j] = (255,255,255)
        else:
            im[i][j] = (0,0,0)
cv.drawKeylines()
cv.imwrite("/home/river-charles/kinova/src/artbot/src/test/output_images/k_means_layered/k_means_colors1_copy.jpg", im, )


