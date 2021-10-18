#!/usr/bin/env python3
###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2021 Kinova inc. All rights reserved.
#
# This software may be modified and distributed 
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import rospy
import time
import math
import random

import actionlib
import PIL
import numpy as np
from PIL import Image
import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os
import glob

from kortex_driver.srv import *
from kortex_driver.msg import *

class ExampleWaypointActionClient:
    # some stupid setup code for the arm that I don't understand
    def __init__(self):
        try:
            rospy.init_node('example_waypoint_action_python')

            self.HOME_ACTION_IDENTIFIER = 2

            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3")
            self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 7)
            self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", False)

            rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))

            # Init the action topic subscriber
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None
            self.position_sub = rospy.Subscriber("/" + self.robot_name + "/base_feedback", BaseCyclic_Feedback, self.cb_position)
            self.client = actionlib.SimpleActionClient('/' + self.robot_name + '/cartesian_trajectory_controller/follow_cartesian_trajectory', kortex_driver.msg.FollowCartesianTrajectoryAction)
            self.client.wait_for_server()

            # Init the services
            clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            read_action_full_name = '/' + self.robot_name + '/base/read_action'
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

            execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

            activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)

            get_product_configuration_full_name = '/' + self.robot_name + '/base/get_product_configuration'
            rospy.wait_for_service(get_product_configuration_full_name)
            self.get_product_configuration = rospy.ServiceProxy(get_product_configuration_full_name, GetProductConfiguration)

            self.raised_z = .055
            self.lowered_z = 0.04578

            self.x_lower = 0.3125
            self.x_upper = self.x_lower + .28
            self.y_lower = -0.0735
            self.y_upper = self.y_lower + .215
            self.paint_refill_z = 0.17
            self.paint_refill_x = 0.433
            self.paint_refill_y = -0.135

            self.list_contours = []

        except:
            self.is_init_success = False
        else:
            self.is_init_success = True
    # I think this is something ros uses
    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event
    # sets the position of the arm to the xyz in the code
    def cb_position(self, position):
        self.current_x = position.base.commanded_tool_pose_x
        self.current_y = position.base.commanded_tool_pose_y
        self.current_z = position.base.commanded_tool_pose_z
    # intellegently selects k representitive colors to use in a palette
    def k_means(self, image, k):
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
    # randomly selects n colors to use in a palette
    def random_palette(self, n):
        palette = []
        for i in range(12):
            palette.append((random.random()*255, random.random()*255, random.random()*255))
        return palette
    # the 12 colors of paint that I purchased, use if too lazy to mix paints
    def default_palette(self):
        return [#(245, 102, 65), #ultramarine blue
                # (1, 22, 207), #spectral red
                (43, 158, 252), #spectral orange
                # (247, 244, 243), #titanium white
                # (154, 164, 0), #turquoise
                (7, 192, 255), #spectral yellow
                (32, 31, 35), #ivory black
                (36,50,127), #burnt umber
                # (48, 235, 38), #spectral green
                # (255, 0, 255), #magenta
                (180,229,255), #peach
                (42,42,165) #brown
                ]
    # selects the geometrically closest color in the palette to use for the given color
    def choose_color(self, color, palette):
        palette_color = palette[0]
        for c in palette:
            if math.dist(c,color) < math.dist(palette_color,color):
                palette_color = c
        return palette_color
    # takes an image file and roughs it up a bit and outputs a list of list of contours
    # each list of contours corresponds to the color at the same index in the palette
    def read_contour(self):
        files = glob.glob('/home/river-charles/kinova/src/artbot/src/test/output_images/k_means_layered/*')
        for f in files:
            os.remove(f)

        im = cv.imread("/home/river-charles/kinova/src/artbot/src/test/input_images/monalisa.jpg")
        scale_percent = 30 # percent of original size
        width = int(im.shape[1] * scale_percent / 100)
        height = int(im.shape[0] * scale_percent / 100)
        dim = (width, height)
        im = cv.resize(im, dim, interpolation = cv.INTER_AREA)

        k = 4
        palette = self.default_palette()
        k = len(palette)
        palette.sort(key=sum)
        grey_palette = []
        for color in palette:
            grey_palette.append(int((color[0]+color[1]+color[2])/3))

        canvases = []
        for i in range(k+1):
            canvas = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
            for row in canvas:
                for cell in row:
                    cell[0] = 255
                    cell[1] = 255
                    cell[2] = 255
            canvases.append(canvas)
        contour_im = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
        for row in contour_im:
                for cell in row:
                    cell[0] = 255
                    cell[1] = 255
                    cell[2] = 255 

        for i in range(len(im)):
            for j in range(len(im[0])):
                color = im[i][j]
                new_color = self.choose_color(color, palette)
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

        ds = 6
        for i in range(len(grey_canvases)):
            layer_contours = []
            for j in range(200):
                if j == 0:
                    dilatation_size = int (ds/2)
                    dilation_shape = cv.MORPH_ELLIPSE
                    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
                    grey_canvases[i] = cv.dilate(grey_canvases[i], element)
                ret, thresh = cv.threshold(grey_canvases[i], 254, 255, cv.THRESH_BINARY_INV)
                contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                if len(contours) == 0:
                    break
                for contour in contours:
                    if len(contour) > 5:
                        layer_contours.append(contour)
                dilatation_size = ds
                dilation_shape = cv.MORPH_ELLIPSE
                element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
                grey_canvases[i] = cv.dilate(grey_canvases[i], element)
            # ds += 1
            self.list_contours.append(layer_contours)


        for i in range(len(self.list_contours)):
            # cv.drawContours(contour_im, self.list_contours[i], -1, palette[0], 1)
            cv.drawContours(contour_im, self.list_contours[i], -1, palette[i], 3)
        cv.imwrite("/home/river-charles/kinova/src/artbot/src/test/output_images/k_means_layered/contours.jpg", contour_im)

        for i in range(k+1):
            cv.imwrite("/home/river-charles/kinova/src/artbot/src/test/output_images/k_means_layered/k_means_colors" + str(i) + ".jpg", canvases[i])
    # helper function to set values for a xyz point in space
    def FillCartesianWaypoint(self, new_x, new_y, new_z, new_theta_x, new_theta_y, new_theta_z, blending_radius):
        cartesianWaypoint = CartesianWaypoint()

        cartesianWaypoint.pose.x = new_x
        cartesianWaypoint.pose.y = new_y
        cartesianWaypoint.pose.z = new_z
        cartesianWaypoint.pose.theta_x = new_theta_x
        cartesianWaypoint.pose.theta_y = new_theta_y
        cartesianWaypoint.pose.theta_z = new_theta_z
        cartesianWaypoint.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE
        cartesianWaypoint.blending_radius = blending_radius
       
        return cartesianWaypoint
    # example code that might be useless
    def example_subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        rospy.sleep(1.0)
        return True
    # example code that might be useless
    def example_clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            rospy.sleep(2.5)
            return True
    # paints all the contours in the correct color cleaning the brush between each color switch
    def drawing_main(self):
        self.read_contour()
        
        current_contour = 0
        start_contour = 90
        for i in range(len(self.list_contours)):
            for contour in self.list_contours[i]:
                if start_contour <= current_contour:
                    self.paint_refill(i)
                    self.draw(contour, i)
                print(current_contour)
                current_contour += 1
            if start_contour <= current_contour:
                self.clean_brush()
    # tells the arm where to go to paint the given contour in the given color
    def draw(self, contour, color):

        goal = FollowCartesianTrajectoryGoal()
        x = self.x_lower + 0.02 + contour[0][0][0] * 0.0003
        y = self.y_upper - 0.02 - contour[0][0][1] * 0.0003
        x1 = self.x_lower + 0.02 + contour[1][0][0] * 0.0003
        y1 = self.y_upper - 0.02 - contour[1][0][1] * 0.0003
        if x == x1 and y1 > y:
            angle = 1.57079
        elif x == x1 and y1 < y:
            angle = -1.57079
        else:
            angle = math.atan((y1-y)/(x1-x))
        goal.trajectory.append(self.FillCartesianWaypoint(x, y, self.paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        if x < x1:
            goal.trajectory.append(self.FillCartesianWaypoint(x-math.cos(angle)*.01, y-math.sin(angle)*.01, self.raised_z, math.radians(180), math.radians(0), math.radians(90), 0))
        else:
            goal.trajectory.append(self.FillCartesianWaypoint(x+math.cos(angle)*.01, y+math.sin(angle)*.01, self.raised_z, math.radians(180), math.radians(0), math.radians(90), 0))
        for i in range(len(contour)-1):
            x = self.x_lower + 0.02 + contour[i][0][0] * 0.0003
            y = self.y_upper - 0.02 - contour[i][0][1] * 0.0003
            x1 = self.x_lower + 0.02 + contour[i+1][0][0] * 0.0003
            y1 = self.y_upper - 0.02 - contour[i+1][0][1] * 0.0003
            if x == x1 and y1 > y:
                angle = 1.57079
            elif x == x1 and y1 < y:
                angle = -1.57079
            else:
                angle = math.atan((y1-y)/(x1-x))
            if x < x1:
                goal.trajectory.append(self.FillCartesianWaypoint(x-math.cos(angle)*.01, y-math.sin(angle)*.01, self.raised_z, math.radians(180), math.radians(0), math.radians(90), 0))
            else:
                goal.trajectory.append(self.FillCartesianWaypoint(x+math.cos(angle)*.01, y+math.sin(angle)*.01, self.raised_z, math.radians(180), math.radians(0), math.radians(90), 0))
            
            if len(goal.trajectory) > 500:
                self.client.send_goal(goal)
                self.client.wait_for_result()
                goal = FollowCartesianTrajectoryGoal()
                self.paint_refill(color)
                goal.trajectory.append(self.FillCartesianWaypoint(x, y, self.paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
                goal.trajectory.append(self.FillCartesianWaypoint(x-math.cos(angle)*.01, y-math.sin(angle)*.01, self.raised_z, math.radians(180), math.radians(0), math.radians(90), 0))
                

        self.client.send_goal(goal)
        self.client.wait_for_result()
    # tells the arm where to go to get more paint of the given color
    def paint_refill(self, paint):
        paint_z = 0.059
        paint_locations = [
            [0.51535, -0.17824], # paint 0
            [0.51215, -0.25797], # paint 1
            [0.51732, -0.34333], # paint 2
            [0.42614, -0.18212], # paint 3
            [0.42614, -0.25797], # paint 4
            [0.42614, -0.34333], # paint 5
        ]
        top_z = 0.175
        paint_x = paint_locations[paint][0]
        paint_y = paint_locations[paint][1]
        paint_z = 0.059

        goal = FollowCartesianTrajectoryGoal()
        goal.trajectory.append(self.FillCartesianWaypoint(self.current_x, self.current_y, top_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(paint_x, paint_y, top_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(paint_x, paint_y, paint_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(paint_x + .01, paint_y, paint_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(paint_x + .01, paint_y + .01, paint_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(paint_x, paint_y + .01, paint_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(paint_x, paint_y, paint_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(paint_x, paint_y, top_z, math.radians(180), math.radians(0), math.radians(90), 0))
        self.client.send_goal(goal)
        self.client.wait_for_result()
    # tells the arm where to go to clean the brush off
    def clean_brush(self):
        water_z = 0.05125
        paint_refill_z = 0.17
        water_x = 0.34751
        water_y = -0.348473

        towel_z = 0.038283
        towel_top_x = 0.251342
        towel_mid_x = 0.2144
        towel_bot_x = 0.19113
        towel_left_y = -0.184
        towel_right_y = -0.36968



        goal = FollowCartesianTrajectoryGoal()
        # get rid of paint
        goal.trajectory.append(self.FillCartesianWaypoint(self.current_x, self.current_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_top_x, towel_left_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_top_x, towel_left_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_top_x, towel_right_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_top_x, towel_left_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_top_x, towel_right_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_top_x, towel_right_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        
        # go in water
        goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        for i in range(3):    
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x + .01, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x + .01, water_y + .01, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y + .01, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        
        # dry off
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_left_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_left_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_right_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_left_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_right_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_right_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))

        # go in water
        goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        for i in range(3):    
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x + .01, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x + .01, water_y + .01, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y + .01, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))

        # dry off
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_left_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_left_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_right_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_left_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_right_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_right_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))

        self.client.send_goal(goal)
        self.client.wait_for_result()
    
    def main(self):
        # For testing purposes
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/waypoint_action_python")
        except:
            pass

        if success:
            self.drawing_main()
        if not success:
            rospy.logerr("The example encountered an error.")


if __name__ == "__main__":
    ex = ExampleWaypointActionClient()
    ex.main()
