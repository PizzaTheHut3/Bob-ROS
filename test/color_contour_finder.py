#!/usr/bin/python3
import cv2 as cv
import random
import numpy
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def k_means(image, k):
    all_colors = []
    for row in image:
        for pixel in row:
            all_colors.append(pixel)
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(all_colors)
    palette = []
    for color in kmeans.cluster_centers_:
        palette.append((int (color[0]), int (color[1]), int (color[2])))
    return palette

def random_palette(n):
    palette = []
    for i in range(12):
        palette.append((random.random()*255, random.random()*255, random.random()*255))
    return palette

def default_palette():
    return [(65, 102, 245), #ultramarine blue
            (207, 22, 1), #spectral red
            (252, 158, 43), #spectral orange
            (243, 244, 247), #titanium white
            (0, 164, 154), #turquoise
            (255, 192, 7), #spectral yellow
            (35, 31, 32), #ivory black
            (127,50,36), #burnt umber
            (38, 235, 48), #spectral green
            (255, 0, 255), #magenta
            (255,229,180), #peach
            (165,42,42) #brown
            ]

def choose_color(color, palette):
    palette_color = palette[0]
    for c in palette:
        if math.dist(c,color) < math.dist(palette_color,color):
            palette_color = c
    return palette_color

im = cv.imread("/home/river-charles/kinova/src/artbot/src/test/input_images/yin_yang.jpeg")
scale_percent = 250 # percent of original size
width = int(im.shape[1] * scale_percent / 100)
height = int(im.shape[0] * scale_percent / 100)
dim = (width, height)
im = cv.resize(im, dim, interpolation = cv.INTER_AREA)
img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
canvas = numpy.zeros((im.shape[0], im.shape[1], 3), numpy.uint8)
for row in canvas:
    for cell in row:
        cell[0] = 255
        cell[1] = 255
        cell[2] = 255

num_contours = 0
num_points = 0

palette = default_palette()

for i in range(20,190,10):
# for i in [150]:
    ret, thresh = cv.threshold(img, i, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    new_contours = []
    for contour in contours:
        longest = 100
        if len(contour) > longest:
            for i in range(0, len(contour), longest):
                num_contours += 1
                new_contours.append(contour[i: min(len(contour), i + longest)])
        elif len(contour) > 100:
            num_contours += 1
            new_contours.append(contour)
    contours = new_contours

    for contour in contours:
        color = im[contour[0][0][1]][contour[0][0][0]]
        # color = choose_color(color, palette)

        # color = (122,35,17) #black
        # color = (255-i,255-i,255-i) #grey by layer
        color = (int (color[0]), int (color[1]), int (color[2])) #color
        for i in range(len(contour)-1):
            num_points += 1
            cv.line(canvas, (contour[i][0][0], contour[i][0][1]),
             (contour[i+1][0][0], contour[i+1][0][1]), color, 7)

print(num_contours, "contours")
print(num_points/390, "minutes")
cv.imwrite("/home/river-charles/kinova/src/artbot/src/test/output_images/contour_recolored.jpg", canvas)


    
