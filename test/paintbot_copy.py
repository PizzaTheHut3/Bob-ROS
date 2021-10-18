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

            self.raised_z = .055
            self.lowered_z = 0.04578

            self.x_lower = 0.3125
            self.x_upper = self.x_lower + .28
            self.y_lower = -0.0735
            self.y_upper = self.y_lower + .215
            self.paint_refill_z = 0.17
            self.paint_refill_x = 0.433
            self.paint_refill_y = -0.135

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
    
    def read_contour(self, image_name):
        im = cv.imread(image_name)

        scale_percent = 250 # percent of original size
        width = int(im.shape[1] * scale_percent / 100)
        height = int(im.shape[0] * scale_percent / 100)
        dim = (width, height)
        im = cv.resize(im, dim, interpolation = cv.INTER_AREA)

        self.image_y = im.shape[0]
        self.image_x = im.shape[1]
        self.list_contours = []

        blank_image = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
        for row in blank_image:
            for cell in row:
                cell[0] = 255
                cell[1] = 255
                cell[2] = 255

        # for i in range(20,190,10):
        for i in [150]:
            img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(img, i, 255, 0)
            # thresh_invert = 255 - thresh

            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            
            for c in contours:
                if len(c) > 100:
                    self.contours.append(c)

            
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

    def drawing_main(self):
        self.read_contour("/home/river-charles/kinova/src/artbot/src/test/input_images/yin_yang.jpeg")
        
        for i in range(len(self.list_contours)):
            for contour in self.list_contours[i]:
                self.paint_refill(i)
                self.draw(contour)

    def draw(self, contour):

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
        goal.trajectory.append(self.FillCartesianWaypoint(x-math.cos(angle)*.01, y-math.sin(angle)*.01, self.raised_z, math.radians(180), math.radians(0), math.radians(90), 0))
        
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
            goal.trajectory.append(self.FillCartesianWaypoint(x, y, self.lowered_z, math.radians(180), math.radians(0), math.radians(90), 0))
            
            if len(goal.trajectory) > 500:
                self.client.send_goal(goal)
                self.client.wait_for_result()
                goal = FollowCartesianTrajectoryGoal()
                self.paint_refill()
                goal.trajectory.append(self.FillCartesianWaypoint(x, y, self.paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
                goal.trajectory.append(self.FillCartesianWaypoint(x-math.cos(angle)*.01, y-math.sin(angle)*.01, self.raised_z, math.radians(180), math.radians(0), math.radians(90), 0))
                

        self.client.send_goal(goal)
        self.client.wait_for_result()
    
    def paint_refill(self, paint):
        paint_z = 0.059
        paint_locations = [
            [0.51535, -0.17824], # paint 0
            [0.51215, -0.25797] # paint 1
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

        # go in water
        goal.trajectory.append(self.FillCartesianWaypoint(self.current_x, self.current_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        for i in range(3):    
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x + .01, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x + .01, water_y + .01, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y + .01, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        
        # dry off
        goal.trajectory.append(self.FillCartesianWaypoint(self.current_x, self.current_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_left_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_left_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_right_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_left_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_mid_x, towel_right_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))

        # go in water
        goal.trajectory.append(self.FillCartesianWaypoint(self.current_x, self.current_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        for i in range(3):    
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x + .01, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x + .01, water_y + .01, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y + .01, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
            goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, water_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(water_x, water_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))

        # dry off
        goal.trajectory.append(self.FillCartesianWaypoint(self.current_x, self.current_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_left_y, paint_refill_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_left_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_right_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_left_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))
        goal.trajectory.append(self.FillCartesianWaypoint(towel_bot_x, towel_right_y, towel_z, math.radians(180), math.radians(0), math.radians(90), 0))

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
