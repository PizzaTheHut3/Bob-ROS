#!/usr/bin/python3
import cv2 as cv
import random
import numpy

im = cv.imread("/home/river-charles/kinova/src/artbot/src/test/input_images/starry.jpg")
scale_percent = 50 # percent of original size
width = int(im.shape[1] * scale_percent / 100)
height = int(im.shape[0] * scale_percent / 100)
dim = (width, height)
im = cv.resize(im, dim, interpolation = cv.INTER_AREA)
blank_image = numpy.zeros((im.shape[0], im.shape[1], 3), numpy.uint8)
img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

num_contours = 0

for i in range(0,250,10):
    ret, thresh = cv.threshold(img, i, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    new_contours = []
    for contour in contours:
        # if len(contour) > 10000:
        #     # new_contours.append(contour)

        #     for i in range(0, len(contour), 100):
        #         num_contours += 1
        #         new_contours.append(contour[i: min(len(contour) - 1, i + 100)])
        if len(contour) > 50:
            num_contours += 1
            new_contours.append(contour)

    contours = new_contours

    count = 0
    for contour in contours:
        color = im[contour[0][0][1]][contour[0][0][0]]
        # cv.drawContours(blank_image, contours, count, (255,255,255), 1)
        cv.drawContours(blank_image, contours, count, (int (color[0]), int (color[1]), int (color[2])), 3)
        count += 1

print(num_contours)
cv.imwrite("/home/river-charles/kinova/src/artbot/src/test/output_images/contour1.jpg", blank_image)

