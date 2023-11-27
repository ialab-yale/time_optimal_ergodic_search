#!/usr/bin/env python3

from tkinter import Y
# import rospy
import rclpy
from rclpy.node import Node

import time
import numpy as np
import matplotlib.pyplot as plt

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper

from multiprocessing.context import ForkContext
from turtle import right
# import rospy
import struct
# import tf
import time
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import jax.numpy as np

# from ergodic_cbf.msg import Cmd
from geometry_msgs.msg import Twist, Pose
# import tf_conversions
import tf2_ros
import geometry_msgs.msg
from std_msgs.msg import Float32MultiArray, Bool

import sys 
sys.path.append('../')

from geometry_msgs.msg import Pose, Twist


class Robot(SyncCrazyflie, Node):
    def __init__(self, link_uri, cf=None):
        SyncCrazyflie.__init__(self, link_uri, cf)
        Node.__init__(self, 'robot')

        self.SIM_FREQ = 240
        self.TIMESTEP = 1./self.SIM_FREQ
        self.AGGR_PHY_STEPS = 1
        self.target_vals = [float(0.25), 
                                float(0.25), 
                                float(0.25), 
                                float(0), 
                                float(0), 
                                float(0)]

    def initialize(self):
        self.initialize_ros()
        self.initialize_logs()

    def initialize_ros(self):
        timer_freq_hz = 60
        timer_period_sec = 1/timer_freq_hz

        #### Declare publishing on 'obs' and create a timer to call 
        #### action_callback every timer_period_sec ################
        self.publisher_ = self.create_publisher(Float32MultiArray, 'obs', 1)
        self.timer = self.create_timer(timer_period_sec, self.step_callback)

        self.publisher_pose = self.create_publisher(Pose, 'pose', 1)

        self.publisher_twist = self.create_publisher(Twist, 'twist', 1)
        #### Subscribe to topic 'action' ###########################
        self.action_subscription = self.create_subscription(Float32MultiArray, 'action', self.get_action_callback, 1)
        self.action_subscription  # prevent unused variable warning

        self.land_subscription = self.create_subscription(Bool, 'land', self.get_land_callback, 1)
        self.land_subscription  # prevent unused variable warning

        self._pose  = Pose()
        self._twist = Twist()


    def initialize_logs(self):
        pose_config = LogConfig(name='position_log', period_in_ms=10)
        pose_config.add_variable('stateEstimate.x' , 'float')
        pose_config.add_variable('stateEstimate.y' , 'float')
        pose_config.add_variable('stateEstimate.z' , 'float')
        pose_config.add_variable('stateEstimate.yaw',   'float')
        pose_config.add_variable('stateEstimate.pitch', 'float')
        pose_config.add_variable('stateEstimate.roll',  'float')
        self.pose_config = pose_config

        twist_config = LogConfig(name='velocityEstimate', period_in_ms=10)
        twist_config.add_variable('stateEstimate.vx', 'float')
        twist_config.add_variable('stateEstimate.vy', 'float')
        twist_config.add_variable('stateEstimate.vz', 'float')
        twist_config.add_variable('gyro.x', 'float')
        twist_config.add_variable('gyro.y', 'float')
        twist_config.add_variable('gyro.z', 'float')
        self.twist_config = twist_config

        # add logging config to the crazyflie
        self.cf.log.add_config(self.pose_config)
        self.cf.log.add_config(self.twist_config)

        # add callbacks
        self.pose_config.data_received_cb.add_callback(self.pose_callback)
        self.twist_config.data_received_cb.add_callback(self.twist_callback)


    ## Callback functions to publish data to ROS
    def step_callback(self):
        msg = Float32MultiArray()
        # msg.data = [data['stateEstimate.x'], data['stateEstimate.x'], data['stateEstimate.x'], \
        #             data['stateEstimate.yaw']*np.pi/180, data['stateEstimate.pitch']*np.pi/180, data['stateEstimate.roll']*np.pi/180, 0, \
        #             data['stateEstimate.vx'], data['stateEstimate.vy'], data['stateEstimate.vz'], \
        #             data['gyro.x']*np.pi/180, data['gyro.y']*np.pi/180, data['gyro.z']*np.pi/180]
        msg.data = [self._pose.position.x, self._pose.position.y, self._pose.position.z, \
                    self._pose.orientation.x, self._pose.orientation.y, self._pose.orientation.z, 0., \
                    0., 0., 0., \
                    self._twist.linear.x, self._twist.linear.y, self._twist.linear.z, \
                    self._twist.angular.x, self._twist.angular.y, self._twist.angular.z]
        self._obs = msg.data
        self.publisher_.publish(msg)

    def pose_callback(self, timestamp, data, logconf):   
        # publish pose
        self._pose.position.x    = float(data['stateEstimate.x'])
        self._pose.position.y    = float(data['stateEstimate.y'])
        self._pose.position.z    = float(data['stateEstimate.z'])
        self._pose.orientation.x = float(data['stateEstimate.yaw']*np.pi/180)
        self._pose.orientation.y = float(-data['stateEstimate.pitch']*np.pi/180)
        self._pose.orientation.z = float(data['stateEstimate.roll']*np.pi/180)
        self._pose.orientation.w = 0.
        self.publisher_pose.publish(self._pose)


    def twist_callback(self, timestamp, data, logconf):   
        self._twist.linear.x = float(data['stateEstimate.vx'])
        self._twist.linear.y = float(data['stateEstimate.vy'])
        self._twist.linear.z = float(data['stateEstimate.vz'])
        self._twist.angular.x = float(data['gyro.x']*np.pi/180)
        self._twist.angular.y = float(data['gyro.y']*np.pi/180)
        self._twist.angular.z = float(data['gyro.z']*np.pi/180)
        self.publisher_twist.publish(self._twist)

        
    # Crazyflie interface commands
    def activate_high_level_commander(self):
        self.cf.param.set_value('commander.enHighLevel', '1')
    
    def take_off(self):
        # self.cf.high_level_commander.takeoff(0.4, 2.0)
        self.cf.commander.send_position_setpoint(0.25, 0.25, 0.4, 0.0)

    def land(self):
        # self.cf.high_level_commander.land(0.0, 2.0)
        self.cf.commander.send_position_setpoint(self._pose.position.x, self._pose.position.y, 0.0, self._pose.orientation.x)

    def go_to_pos(self, x, y, yaw=0.0):
        # self.cf.high_level_commander.go_to(x, y, 0.4, 0.0, 2.0, relative=False)
        self.cf.commander.send_position_setpoint(x, y, 0.4, yaw)

    def go_to_vel(self, vx, vy, yawrate=0.0):
        self.cf.commander.send_hover_setpoint(vx, vy, yawrate, 0.4) # TODO try send_vel_world_setpoint

    def start_config(self):
        print(self.pose_config.cf)
        self.pose_config.start()
        self.twist_config.start()

    def stop_config(self):
        self.pose_config.stop()
        self.twist_config.stop()

    #### Read the action to apply to the env from topic 'action'
    def get_action_callback(self, msg):
        self.target_vals = msg.data
        self.go_to_pos(self.target_vals[0], self.target_vals[1])

    def get_land_callback(self, msg):
        if msg.data:
            self.land()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    # URI to the Crazyflie to connect to
    URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E700')

    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    with Robot(URI, cf=Crazyflie(rw_cache='./cache')) as robot:
        robot.initialize()
        # cf_interface.activate_high_level_commander()
        # cf_interface.wait_for_param_download()
        
        # cf_interface.take_off()
        robot.start_config()
        rclpy.spin(robot)
        robot.land()
        
        robot.stop_config()
    

if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     rospy.init_node('drone_node')
    
#     # URI to the Crazyflie to connect to
#     URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E701')

#     # Initialize the low-level drivers
#     cflib.crtp.init_drivers()

#     rate = rospy.Rate(60)
#     with Robot(URI, cf=Crazyflie(rw_cache='./cache')) as robot:
#         robot.initialize()
#         # cf_interface.activate_high_level_commander()
#         # cf_interface.wait_for_param_download()
        
#         # cf_interface.take_off()
#         robot.start_config()
#         try:
#             while not rospy.is_shutdown():
#                 if robot.last_cmd.type == 0:
#                     robot.go_to_pos(robot.last_cmd.x, robot.last_cmd.y)
#                 else:
#                     robot.go_to_vel(robot.last_cmd.x, robot.last_cmd.y)
                
#                 # robot.go_to_vel(robot.last_cmd.x, robot.last_cmd.y)
#                 # robot.go_to_pos(robot.last_cmd.x, robot.last_cmd.y)
#                 rate.sleep()
#         except KeyboardInterrupt:
#             robot.land()
        
#         robot.stop_config()