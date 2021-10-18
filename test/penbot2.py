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
from sklearn.cluster import KMeans
import numpy

from kortex_driver.srv import *
from kortex_driver.msg import *

class ExampleWaypointActionClient:
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

            self.raised_z = 0.027
            self.lowered_z = 0.023
            self.force_z = 0

            self.x_upper = 0.63 # .63 - .35 = .28
            self.x_lower = 0.353
            self.y_upper = 0.125 # .125 + .09 = .215
            self.y_lower = -0.087
        except:
            self.is_init_success = False
        else:
            self.is_init_success = True

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event
    
    def cb_position(self, position):
        self.current_x = position.base.commanded_tool_pose_x
        self.current_y = position.base.commanded_tool_pose_y
        self.current_z = position.base.commanded_tool_pose_z
    
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

    def random_palette(self, n):
        palette = []
        for i in range(12):
            palette.append((random.random()*255, random.random()*255, random.random()*255))
        return palette

    def default_palette(self):
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

    def choose_color(self, color, palette):
        palette_color = palette[0]
        for c in palette:
            if math.dist(c,color) < math.dist(palette_color,color):
                palette_color = c
        return palette_color
    
    def read_contour(self, image_name):
        im = cv.imread(image_name)

        scale_percent = 40 # percent of original size
        width = int(im.shape[1] * scale_percent / 100)
        height = int(im.shape[0] * scale_percent / 100)
        dim = (width, height)
        im = cv.resize(im, dim, interpolation = cv.INTER_AREA)

        self.image_y = im.shape[0]
        self.image_x = im.shape[1]
        self.contours = []

        blank_image = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
        for row in blank_image:
            for cell in row:
                cell[0] = 255
                cell[1] = 255
                cell[2] = 255
        k = 8
        palette = self.k_means(im, k)

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

        all_contours = []
        ds = 3
        for grey_canvas in grey_canvases:
            layer_contours = []
            for i in range(100):
                ret, thresh = cv.threshold(grey_canvas, 254, 255, cv.THRESH_BINARY_INV)
                contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                if len(contours) == 0:
                    break
                for contour in contours:
                    layer_contours.append(contour)
                dilatation_size = ds
                dilation_shape = cv.MORPH_ELLIPSE
                element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
                grey_canvas = cv.dilate(grey_canvas, element)
            ds += 1
            all_contours.append(layer_contours)


        for i in range(len(all_contours)):
            cv.drawContours(contour_im, all_contours[i], -1, palette[0], 1)
            for contour in all_contours[i]:
                self.contours.append(contour)
            #  cv.drawContours(contour_im, all_contours[i], -1, palette[i], 3)

        for i in range(k+1):
            cv.imwrite("/home/river-charles/kinova/src/artbot/src/test/output_images/k_means_layered/k_means_colors" + str(i) + ".jpg", canvases[i])
            cv.drawContours(blank_image, self.contours, -1, (0, 0, 0), 1)

        cv.imwrite("/home/river-charles/kinova/src/artbot/src/test/output_images/noworky.jpg", blank_image)

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

    def callback(self, data):
        self.force_z = data.base.tool_external_wrench_force_z

    def drawing_main(self):
        self.read_contour("/home/river-charles/kinova/src/artbot/src/test/input_images/vangogh.jpg")
        
        rospy.Subscriber('/my_gen3/base_feedback', kortex_driver.msg.BaseCyclic_Feedback, self.callback)

        for contour in self.contours:
            self.draw(contour)

    def draw(self, contour):

        # delta_x = (self.x_upper - self.x_lower) * scale / 2
        # x_scale_upper = self.x_upper - delta_x
        # x_scale_lower = self.x_lower + delta_x

        # delta_y = (self.y_upper - self.y_lower) * scale / 2
        # y_scale_upper = self.y_upper - delta_y
        # y_scale_lower = self.y_lower + delta_y

        goal = FollowCartesianTrajectoryGoal()
        start_point = contour[0]

        x = self.x_lower + 0.02 + start_point[0][0] * 0.0003
        y = self.y_upper - 0.02 - start_point[0][1] * 0.0003

        goal.trajectory.append(self.FillCartesianWaypoint(x, y, self.raised_z, math.radians(180), math.radians(0), math.radians(90), 0))
        self.client.send_goal(goal)
        self.client.wait_for_result()
        
        current_z = self.raised_z
        for i in range(int (self.raised_z*1000), int (self.lowered_z*1000), -1):
            goal = FollowCartesianTrajectoryGoal()
            current_z = (float (i))/1000
            goal.trajectory.append(self.FillCartesianWaypoint(x, y, (float (i))/1000, math.radians(180), math.radians(0), math.radians(90), 0))
            self.client.send_goal(goal)
            self.client.wait_for_result()
            if self.force_z > 13:
                break

        for i in contour:
            x = self.x_lower + 0.02 + i[0][0] * 0.0003
            y = self.y_upper - 0.02 - i[0][1] * 0.0003
            goal.trajectory.append(self.FillCartesianWaypoint(x, y, current_z, math.radians(180), math.radians(0), math.radians(90), 0))
            if len(goal.trajectory) > 200:
                self.client.send_goal(goal)
                self.client.wait_for_result()
                goal = FollowCartesianTrajectoryGoal()
                if self.force_z > 15:
                    current_z = current_z + .001
                if self.force_z < 10:
                    current_z = current_z - .001
        
        goal.trajectory.append(self.FillCartesianWaypoint(x, y, self.raised_z, math.radians(180), math.radians(0), math.radians(90), 0))

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
