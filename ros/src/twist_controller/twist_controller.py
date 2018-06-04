import rospy
from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PIDController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

LOW_PASS_FREQ = 0.5
INTEGRAL_PERIOD_SEC = 3.0
MIN_TIME_DELTA = 0.0001

class Controller(object):
    def __init__(
        self,
        dbw_parameters, # Set of base parameters
        ):

        self.is_initialized = False
        self.last_time = rospy.get_time()

        self.dbw_parameters = dbw_parameters
        self.steering_controller = YawController(
            wheel_base = self.dbw_parameters['wheel_base'],
            steer_ratio = self.dbw_parameters['steer_ratio'],
            min_speed = self.dbw_parameters['min_vehicle_speed'],
            max_lat_accel = self.dbw_parameters['max_lat_accel'],
            max_steer_angle = self.dbw_parameters['max_steer_angle'])
        self.low_pass_filter_vel = LowPassFilter(samples_num = LOW_PASS_FREQ * self.dbw_parameters['dbw_rate'])
        self.throttle_controller = PIDController(
            kp = 0.3,
            ki = 0.01,
            kd = 0.01,
            mn = self.dbw_parameters['vehicle_min_throttle_val'],
            mx = self.dbw_parameters['vehicle_max_throttle_val'],
            int_buffer_size = INTEGRAL_PERIOD_SEC * self.dbw_parameters['dbw_rate'])

    def control(
        self,
        target_dx,
        target_dyaw,
        current_dx,
        current_dyaw,
        dbw_status = True):

        throttle = 0
        brake = 0
        steering = 0
        correct_data = False

        cur_time = rospy.get_time();

        current_dx_smooth = self.low_pass_filter_vel.filt(current_dx)

        if not dbw_status:
            self.is_initialized = False
        else:
            if not self.is_initialized:
                self.reset()
                self.is_initialized = True

            dv = target_dx - current_dx
            dt = cur_time - self.last_time

            if dt >= MIN_TIME_DELTA:
                steering = self.steering_controller.get_steering(target_dx, target_dyaw, current_dx_smooth)
                throttle = self.throttle_controller.step(dv, dt)

                if target_dx <= current_dx_smooth and current_dx_smooth <= self.dbw_parameters['min_vehicle_speed']:
                    throttle = 0.0
                    brake = self.dbw_parameters['min_vehicle_break_torque']
                #elif dv < 0 and throttle <= self.dbw_parameters['vehicle_throttle_dead_zone']:
                #    throttle = self.dbw_parameters['vehicle_min_throttle_val']
                #    deceleration = min(abs(dv / dt), abs(self.dbw_parameters['decel_limit']))
                #    brake = self.dbw_parameters['vehicle_mass'] * self.dbw_parameters['wheel_radius'] * deceleration
                elif throttle <= self.dbw_parameters['vehicle_throttle_dead_zone']:
                    deceleration = abs(self.dbw_parameters['decel_limit'] * (throttle - self.dbw_parameters['vehicle_throttle_dead_zone']) / (self.dbw_parameters['vehicle_throttle_dead_zone'] - self.dbw_parameters['vehicle_min_throttle_val']))
                    throttle = 0.0
                    brake = self.dbw_parameters['vehicle_mass'] * self.dbw_parameters['wheel_radius'] * deceleration
                
                correct_data = True

        self.last_time = cur_time

        return throttle, brake, steering, correct_data

    def reset(self):
        self.steering_controller.reset()
        self.throttle_controller.reset()
