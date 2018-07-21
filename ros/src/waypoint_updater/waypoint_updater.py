#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
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
MIN_VELOCITY_VALUE = 2.0 # Minimal velocity to be equal to 0.

STOP_WAYPOINT_PASS_LIMIT = 3 # Maximal number to waypoints bassed behind stop line to recognize it as passed
DECELERATION_LOOKAHEAD_WPS_MIN = 20 # Minimal number of waypoint ahead to start plan deceleration
DECELERATION_LOOKAHEAD_ALFA = 0.35 # Define minimal distance to start calc deceleration, in dependance from vehicle velocity
DECELERATION_POWER_SLOW = 0.3 # Define function of deceleration
DECELERATION_POWER_NORM = 2.0 # Define function of deceleration
DECELERATION_POWER_FAST = 3.5 # Define function of deceleration
DECELERATION_STOP_LAG = 2.5 # Approximately equal to half vehicle size. Necessary to stop just before stop-line

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.is_initialized = False
        self.base_waypoints = None
        self.base_waypoints_kdtree = None
        self.hero_position = None
        self.hero_velocity = None

        self.last_stop_idx = -1
        self.last_stop_from_distance = -1
        self.last_stop_from_vel = -1
        self.decel_pow = DECELERATION_POWER_SLOW

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
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
            if self.is_initialized and bool(self.hero_position) and bool(self.hero_velocity):
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

    def prepare_lane(self, from_idx):
        upd_lane = Lane()
        upd_lane.waypoints = []

        if from_idx < len(self.base_waypoints):
            stop_idx = self.stopline_waypoint_idx
            calc_vel = self.base_waypoints[from_idx].twist.twist.linear.x

            if stop_idx >= 0:
                if self.last_stop_idx == stop_idx:
                    calc_vel = self.last_stop_from_vel
            
            min_decel_dist = int(DECELERATION_LOOKAHEAD_ALFA * (calc_vel ** 1.5))
            cur_dist = self.distance(self.base_waypoints, from_idx, stop_idx)

            if (self.stopline_waypoint_idx < 0) or ((from_idx - self.stopline_waypoint_idx) >= STOP_WAYPOINT_PASS_LIMIT) or ((cur_dist > min_decel_dist) and ((stop_idx - from_idx) > DECELERATION_LOOKAHEAD_WPS_MIN)):
                upd_lane.waypoints = self.base_waypoints[from_idx:(from_idx + LOOKAHEAD_WPS)]

                if (from_idx - self.stopline_waypoint_idx) >= STOP_WAYPOINT_PASS_LIMIT:
                    self.last_stop_idx = -1
            else:
                upd_lane.waypoints = self.decelerate(from_idx, self.stopline_waypoint_idx - 1)

        return upd_lane

    def _velocity_calc(self, dist, pow):
        return ((self.last_stop_from_vel - MIN_VELOCITY_VALUE) * max(0, dist / self.last_stop_from_distance)**pow + MIN_VELOCITY_VALUE)

    #added by arm041
    def decelerate(self, from_idx, stop_idx):
        """
        This function will create a lane if a traffic light is detected
        """

        cur_vel = self.hero_velocity.twist.linear.x
        cur_dist = max(0, self.distance(self.base_waypoints, from_idx, stop_idx) - DECELERATION_STOP_LAG)
        to_idx = max(from_idx, stop_idx)

        if self.last_stop_idx != stop_idx:
            self.last_stop_idx = stop_idx
            self.last_stop_from_distance = cur_dist
            self.last_stop_from_vel = self.base_waypoints[from_idx].twist.twist.linear.x
            self.decel_pow = DECELERATION_POWER_SLOW

        if self.last_stop_from_distance > 0:
            if cur_vel > self._velocity_calc(cur_dist, DECELERATION_POWER_NORM):
                self.decel_pow = DECELERATION_POWER_FAST
            elif cur_vel < self._velocity_calc(cur_dist, DECELERATION_POWER_FAST):
                self.decel_pow = DECELERATION_POWER_SLOW

        temp = []
        for idx in range(from_idx, to_idx + 1):
            wp = self.base_waypoints[idx]
            
            dist = max(0, self.distance(self.base_waypoints, idx, stop_idx) - DECELERATION_STOP_LAG)

            if (dist > 0) and (self.last_stop_from_distance > 0):
                vel = self._velocity_calc(dist, self.decel_pow)

                if vel > wp.twist.twist.linear.x:
                    vel = wp.twist.twist.linear.x
            else:
                vel = 0.0
            
            way_point_temp = Waypoint()
            way_point_temp.pose.pose.position.x = wp.pose.pose.position.x
            way_point_temp.pose.pose.position.y = wp.pose.pose.position.y
            way_point_temp.pose.pose.position.z = wp.pose.pose.position.z
            way_point_temp.pose.pose.orientation = wp.pose.pose.orientation
            way_point_temp.twist.twist.linear.x = vel

            temp.append(way_point_temp)
        return temp

    #end by arm041

    def pose_cb(self, pose):
        self.hero_position = pose

    def current_velocity_cb(self, twist):
        self.hero_velocity = twist

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

    def distance(self, waypoints, from_idx, to_idx):
        if from_idx >= to_idx:
            return 0

        dl = lambda a, b: math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2  + (a.z - b.z)**2)
        
        dist = 0
        prev_idx = from_idx
        for idx in range(from_idx + 1, to_idx + 1): 
            dist += dl(waypoints[prev_idx].pose.pose.position, waypoints[idx].pose.pose.position)
            prev_idx = idx

        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
