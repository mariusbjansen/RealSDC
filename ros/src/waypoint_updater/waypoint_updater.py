#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight
from std_msgs.msg import Int32
import math

from scipy.spatial import KDTree

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

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
FINAL_WP_PUBLISH_FREQ = 50 # Frequency of publishing of final waypoints
MAX_DECEL = .3 #The maximum deceleration for the vehicle when trying to break before red light

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.is_initialized = False
        self.base_waypoints = None
        self.base_waypoints_kdtree = None
        self.hero_position = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb) #arm041

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # Add other member variables you need below 
	    #Added by arm041
        self.stopline_waypoint_idx = -1
	    #End arm041

        self._main_cycle()

    def _main_cycle(self):
        rate = rospy.Rate(FINAL_WP_PUBLISH_FREQ)
        while not rospy.is_shutdown():
            if self.is_initialized and bool(self.hero_position):
                next_wp_idx = self.get_next_waypoint_idx(self.hero_position.pose.position.x, self.hero_position.pose.position.y)
                upd_lane = self.prepare_lane(next_wp_idx)
                self.publish_lane(upd_lane)
                
            rate.sleep()

    def publish_lane(self, lane):
        self.final_waypoints_pub.publish(lane)

    def get_next_waypoint_idx(self, x, y):
        """
        Return nearest next waypoint from the list.
        """

        dist, idx = self.base_waypoints_kdtree.query([x, y])

        waypoints_num = len(self.base_waypoints)
        if waypoints_num <= 1:
            return -1

        cur_waypoint = self.base_waypoints[idx]
        cur_dx, cur_dy = (x - cur_waypoint.pose.pose.position.x, y - cur_waypoint.pose.pose.position.y)

        if (idx + 1) < waypoints_num:
            next_waypoint = self.base_waypoints[idx + 1]
            next_dx, next_dy = (x - next_waypoint.pose.pose.position.x, y - next_waypoint.pose.pose.position.y)
            if (next_dx * cur_dx + next_dy * cur_dy) <= 0:
                return idx + 1
            else:
                return idx

        if idx >= 1:
            prev_waypoint = self.base_waypoints[idx - 1]
            prev_dx, prev_dy = (x - prev_waypoint.pose.pose.position.x, y - prev_waypoint.pose.pose.position.y)
            if (prev_dx * cur_dx + prev_dy * cur_dy) < 0:
                return idx

        return -1

    def prepare_lane(self, start_idx):

        upd_lane = Lane()
        temp = []

        waypoints_follow = self.base_waypoints[start_idx:start_idx + LOOKAHEAD_WPS]

        #check added by arm041 to consider the possibility of red light when creating waypoints
        if self.stopline_waypoint_idx == -1 or (self.stopline_waypoint_idx>= start_idx +  LOOKAHEAD_WPS):
            upd_lane.waypoints = waypoints_follow
        else:
            upd_lane.waypoints = self.decelerate(start_idx, waypoints_follow) 
            #temp = self.decelerate(start_idx, self.base_waypoints)
            #for waypoint in temp:
             #   upd_lane.waypoints.append(waypoint)

        return upd_lane

    #added by arm041
    def decelerate (self, start_idx, waypoints_in):
        """
        This function will create a lane if a traffic light is detected
        """
        temp = []
        for i, wp in enumerate(waypoints_in):
            way_point_temp = Waypoint()
            way_point_temp.pose = wp.pose

            stop_idx = max(self.stopline_waypoint_idx - start_idx - 2, 0)
            dist = self.distance (waypoints_in, i, stop_idx)

            vel = math.sqrt (2 * MAX_DECEL * dist)
            if vel < 1: 
                vel = 0
        
            way_point_temp.twist.twist.linear.x = min (vel, wp.twist.twist.linear.x)
            temp.append(way_point_temp)
        return temp

    #end by arm041

    def pose_cb(self, pose):
        self.hero_position = pose

    def waypoints_cb(self, waypoints):
        # Normally is called once, but can be reinitialized if necessary
        self.is_initialized = False        
        
        self.base_waypoints = waypoints.waypoints
        base_waypoints_spatial = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.base_waypoints]
        self.base_waypoints_kdtree = KDTree(base_waypoints_spatial)

        self.is_initialized = True

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message. Implement
	    self.stopline_waypoint_idx = msg.data #arm041
        

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

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
