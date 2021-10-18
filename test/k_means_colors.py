#!/usr/bin/python3
import cv2 as cv
import random
import numpy
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os
import glob

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

files = glob.glob('/home/river-charles/kinova/src/artbot/src/test/output_images/k_means_layered/*')
for f in files:
    os.remove(f)

im = cv.imread("/home/river-charles/kinova/src/artbot/src/test/input_images/boldandbrash.jpg")
scale_percent = 50 # percent of original size
width = int(im.shape[1] * scale_percent / 100)
height = int(im.shape[0] * scale_percent / 100)
dim = (width, height)
im = cv.resize(im, dim, interpolation = cv.INTER_AREA)

k = 5
palette = k_means(im, k)
palette.sort(key=sum)
grey_palette = []
for color in palette:
    grey_palette.append(int((color[0]+color[1]+color[2])/3))

canvases = []
for i in range(k+1):
    canvas = numpy.zeros((im.shape[0], im.shape[1], 3), numpy.uint8)
    for row in canvas:
        for cell in row:
            cell[0] = 255
            cell[1] = 255
            cell[2] = 255
    canvases.append(canvas)
contour_im = numpy.zeros((im.shape[0], im.shape[1], 3), numpy.uint8)
for row in contour_im:
        for cell in row:
            cell[0] = 255
            cell[1] = 255
            cell[2] = 255 

for i in range(len(im)):
    for j in range(len(im[0])):
        color = im[i][j]
        new_color = choose_color(color, palette)
        canvases[0][i][j] = new_color
        canvases[palette.index(new_color) + 1][i][j] = new_color

for i in range(len(canvases)):
    dilatation_size = 2
    dilation_shape = cv.MORPH_RECT
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
    canvases[i] = cv.erode(canvases[i], element)
    canvases[i] = cv.dilate(canvases[i], element)
    canvases[i] = cv.dilate(canvases[i], element)
    canvases[i] = cv.erode(canvases[i], element)

grey_canvases = []
for i in range(1, len(canvases), 1):
    grey = cv.cvtColor(canvases[i], cv.COLOR_BGR2GRAY)
    grey_canvases.append(grey)

grey_count = 0
all_contours = []
ds = 2
contours_copy = []
for grey_canvas in grey_canvases:
    contours_layer = []
    while True:
        cv.imwrite("/home/river-charles/kinova/src/artbot/src/test/output_images/k_means_layered/grey" + str(grey_count) + ".jpg", grey_canvas)
        grey_count += 1
        ret, thresh = cv.threshold(grey_canvas, 254, 255, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            break
        elif len(contours) == 1:
            contours = np.array(contours, dtype='int32')
            contours_copy[0] = contours
            print(np.shape(contours_copy))
            contours_layer = np.concatenate((contours_layer, contours_copy))
        else:
            # contours = np.array(contours)
            contours_copy = contours
            # print(contours[0])
            contours_layer = np.concatenate((contours_layer, contours))
        dilatation_size = ds
        dilation_shape = cv.MORPH_ELLIPSE
        element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
        grey_canvas = cv.dilate(grey_canvas, element)
    ds += 1
    all_contours.append(contours_layer)


count = 0
for i in range(len(all_contours)):
# for i in [4,5]:
    j = 0
    for contour in all_contours[i]:
        if len(contour) > 10:
            cv.drawContours(contour_im, all_contours[i], j, palette[0], 1)
            count += 1 
        j += 1
cv.imwrite("/home/river-charles/kinova/src/artbot/src/test/output_images/k_means_layered/contours.jpg", contour_im)
print(count)

for i in range(k+1):
    cv.imwrite("/home/river-charles/kinova/src/artbot/src/test/output_images/k_means_layered/k_means_colors" + str(i) + ".jpg", canvases[i])



