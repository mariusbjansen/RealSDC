#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import math
import tf
import cv2
import yaml

from scipy.spatial import KDTree

#import ptvsd
#ptvsd.enable_attach("my_secret", address = ('127.0.0.1', 3000))
#ptvsd.wait_for_attach()

TL_DEBUG = True
STATE_COUNT_THRESHOLD = 3

MIN_TL_VISIBLE_DISTANCE = 1 # Minimal TL distance when it's can be visible
MAX_TL_VISIBLE_DISTANCE = 300 # MAximal TL distance when it's can be visible
MAX_STOPLINE_DISTANCE = 50 # Maximal distance between stopline and traffic light
NORMAL_STOPLINE_DISTANCE = 15 # Normal distance between stopline and traffic light

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.is_initialized = False
        self.base_waypoints = None
        self.base_waypoints_kdtree = None

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # Normally is called once, but can be reinitialized if necessary
        self.is_initialized = False        
        
        self.base_waypoints = waypoints.waypoints
        base_waypoints_spatial = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.base_waypoints]
        self.base_waypoints_kdtree = KDTree(base_waypoints_spatial)

        self.is_initialized = True

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera


        """

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint_idx(self, x, y, find_next):
        """
        Return nearest waypoint from the list.
        """

        dist, idx = self.base_waypoints_kdtree.query([x, y])

        waypoints_num = len(self.base_waypoints)
        if waypoints_num <= 1:
            return -1

        cur_waypoint = self.base_waypoints[idx]
        cur_dx, cur_dy = (x - cur_waypoint.pose.pose.position.x, y - cur_waypoint.pose.pose.position.y)
        
        next_idx = idx + 1
        if next_idx >= waypoints_num:
            return -1

        next_waypoint = self.base_waypoints[next_idx]
        next_dx, next_dy = (x - next_waypoint.pose.pose.position.x, y - next_waypoint.pose.pose.position.y)

        if (next_dx * cur_dx + next_dy * cur_dy) <= 0:
            if find_next:
                return next_idx
            else:
                return idx
        else:
            if find_next:
                return idx
            elif (idx - 1) >= 0:
                return idx - 1
            else:
                return idx

        return -1

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        cur_tl_state = self.light_classifier.get_classification(cv_image)

        # TODO: Remove when classifier to be implemented
        if cur_tl_state == TrafficLight.UNKNOWN:
            cur_tl_state = light.state

        return cur_tl_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        closest_light = None
        light_wp_idx = -1
        line_wp_idx = -1

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        
        if bool(self.pose):
            # Waypoint corresponding to car position 
            # Necessary to check if traffic light is ahead or behind
            car_wp_idx = self.get_closest_waypoint_idx(self.pose.pose.position.x, self.pose.pose.position.y, find_next = True)

            if car_wp_idx >= 0:
                # Find the closest visible traffic light (if one exists)
                for i, light in enumerate(self.lights):
                    # Get closest waypoint to TL
                    tl_dist = ((light.pose.pose.position.x - self.pose.pose.position.x) ** 2 + (light.pose.pose.position.y - self.pose.pose.position.y) ** 2) ** 0.5
                    if (tl_dist >= MIN_TL_VISIBLE_DISTANCE) and (tl_dist <= MAX_TL_VISIBLE_DISTANCE):
                        wp_idx = self.get_closest_waypoint_idx(light.pose.pose.position.x, light.pose.pose.position.y, find_next = False)
                        if (wp_idx >= car_wp_idx) and ((light_wp_idx < 0) or (wp_idx < light_wp_idx)):
                            light_wp_idx = wp_idx
                            closest_light = light

                if light_wp_idx >= 0:
                    # Get closest stop line waypoint index
                    for i, stop_line in enumerate(stop_line_positions):
                        tl_dist = ((closest_light.pose.pose.position.x - stop_line[0]) ** 2 + (closest_light.pose.pose.position.y - stop_line[1]) ** 2) ** 0.5
                        if tl_dist <= MAX_STOPLINE_DISTANCE:
                            wp_idx = self.get_closest_waypoint_idx(stop_line[0], stop_line[1], find_next = False)
                            if (wp_idx <= light_wp_idx) and ((line_wp_idx < 0) or (wp_idx > line_wp_idx)):
                                line_wp_idx = wp_idx
                            
                    # Stop line could be missed, in this case assume it's on some usual distance from traffic ligts
                    if line_wp_idx < 0:
                        line_wp_idx = light_wp_idx
                        while line_wp_idx > 0:
                            cur_waypoint = self.base_waypoints[line_wp_idx - 1]
                            tl_dist = ((closest_light.pose.pose.position.x - cur_waypoint.pose.pose.position.x) ** 2 + (closest_light.pose.pose.position.y - cur_waypoint.pose.pose.position.y) ** 2) ** 0.5
                            if tl_dist > NORMAL_STOPLINE_DISTANCE:
                                break
                            line_wp_idx = line_wp_idx - 1

            if line_wp_idx >= 0:
                state = self.get_light_state(closest_light)

                if TL_DEBUG:
                    if state == TrafficLight.UNKNOWN:
                        rospy.loginfo("No active traffic light visible.")
                    else:
                        state_name = "RED" if state == TrafficLight.RED else "GREEN" if state == TrafficLight.GREEN else "YELLOW"
                        tl_dist = ((closest_light.pose.pose.position.x - self.pose.pose.position.x) ** 2 + (closest_light.pose.pose.position.y - self.pose.pose.position.y) ** 2) ** 0.5
                        sl_dist = ((self.base_waypoints[line_wp_idx].pose.pose.position.x - self.pose.pose.position.x) ** 2 + (self.base_waypoints[line_wp_idx].pose.pose.position.y - self.pose.pose.position.y) ** 2) ** 0.5
                        if car_wp_idx > line_wp_idx:
                            sl_dist = -sl_dist

                        rospy.loginfo("{} in {}m, STOP in {}m".format(state_name, tl_dist, sl_dist))

                return line_wp_idx, state

        if TL_DEBUG:
            rospy.loginfo("No active traffic light visible.")

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
