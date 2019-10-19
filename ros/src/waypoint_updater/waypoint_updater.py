#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import numpy as np
import copy

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.curr_vel)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        # other member variables
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopLineWpIdx = -1
        self.currVel = None
        self.decLimit = rospy.get_param('~decel_limit', -5)
        self.accLimit = rospy.get_param('~accel_limit', 1.)
        print("START LOOOP")
        self.loop()

    def loop(self):
        rate = rospy.Rate(31) #31Hz because line follower is running at 30Hz
        while not rospy.is_shutdown():
            if self.pose and self.waypoint_tree:
                #Get closest waypoint
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]
        # Check if closest point is in front of vehicle
        cl_vect = np.array(self.waypoints_2d[closest_idx])
        prev_vect = np.array(self.waypoints_2d[closest_idx - 1])
        pos_vect = np.array([x,y])
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if val > 0:
            # Closest point is behind vehicle take next one
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self, closest_idx):
        lane = self.generate_lane()
        self.final_waypoints_pub.publish(lane)

    def generate_lane(self):
        lane = Lane()
        closestIdx = self.get_closest_waypoint_idx()
        endIdx = closestIdx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closestIdx:endIdx]
        if self.stopLineWpIdx == -1 or self.stopLineWpIdx >= endIdx:
            lane.waypoints = base_waypoints
        else:
            print("dec")
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closestIdx)

        return lane

    def decelerate_waypoints(self, waypoints, closestIdx):
        neededDec = 0
        currVel = self.currVel
        dist = 0
        # 2 is to stop infront of the line
        stopIdx = max(self.stopLineWpIdx - closestIdx - 2, 0)
        #check if we can stop with maximum deceleration
        if closestIdx > 0:
            dist = self.distance(waypoints, 0, stopIdx)
        if dist > 0:
            neededDec = -(currVel * currVel) / (2 * dist)
        else:
            neededDec = self.decLimit - 1.0 # not possible to decelerate anymore
        # check if car is already stopped
        if currVel < 1.0 and stopIdx < 1:
            neededDec = 0
        if neededDec >= self.decLimit:
            newWpList = copy.deepcopy(waypoints)
            #print("newList")
            for i in range(len(newWpList)):
                if stopIdx - i > 1:
                    dist = self.distance(newWpList, i, stopIdx)
                    vel = math.sqrt(2* (-neededDec) * dist)
                    if vel < 1.0:
                        vel = 5.0
                else:
                    vel = 0
                #print(vel)
                self.set_waypoint_velocity(newWpList, i, vel)

            return newWpList
        
        # We cannot break anymore continue
        print("Continue")
        return waypoints
                
    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            print("waypoints")
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message. Implement
        self.stopLineWpIdx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def curr_vel(self, Speed3d):
        self.currVel = Speed3d.twist.linear.x

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
