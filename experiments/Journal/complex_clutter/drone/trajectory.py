import rclpy
from rclpy.node import Node

import logging
import time
import random
import os
import glob
import pickle

import jax.numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu') # run using cpu rather than gpu
import numpy as onp
import matplotlib.pyplot as plt

# from ergodic_cbf_ros.msg import Cmd
from geometry_msgs.msg import Pose, Twist, Point
from std_msgs.msg import Float32MultiArray, Bool
from matplotlib.patches import Rectangle, Circle

import sys
# sys.path.append('../')

# from utils.fn_point_mass_dynamics import Dynamics
# from utils.mpc import MPC
# from utils.problem_setup import ErgProblem
# import utils.fn_cbf as cbf
# from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
# from gym_pybullet_drones.utils.enums import DroneModel, Physics
# from utils.fn_aug_lag_opt import Aug_Lagrange_optimizer


class Computation(Node):
    def __init__(self, freq: int=50, aggregate_phy_steps: int=1):
        super().__init__('computation_node')

        self.xf = np.array([1.75, -0.75])
        self.init_state = np.array([2.0, 3.25])

        step = 1e-3
        ineq = 0.4
        eq = 0.5
        erg = 5.0
        c = 1.0
        iter = 200000
        a = 0.5

        # self.dynam   = Dynamics()
        # self.prob = ErgProblem(self.dynam, self.xf, a)
        # x0 = 0.001*np.zeros((self.prob.T, self.dynam.n))
        # x0 = np.linspace(self.init_state,self.xf,self.prob.T,endpoint=True)
        # u0 = 0.001*np.zeros((self.prob.T, self.dynam.m))
        # z0 = np.concatenate([x0, u0], axis=1)

        # problem = Aug_Lagrange_optimizer(z0, self.init_state, self.prob, self.prob.erg_met, self.prob.f, 
                                        # self.prob.g1, self.prob.g2, self.prob.g3, self.prob.g4, self.prob.g5, self.prob.g6, 
                                        # self.prob.g7, self.prob.g8, self.prob.g9, self.prob.g10, self.prob.g11, self.prob.g12, 
                                        # self.prob.g13, eq=eq, erg=erg, step_size=step, ineq=ineq, c=c)
        # problem = Aug_Lagrange_optimizer(z0, self.init_state, self.prob, self.prob.erg_met, self.prob.f, 
        #                                 self.prob.g1, self.prob.g2, self.prob.g3, self.prob.g4, self.prob.g5, self.prob.g6,
        #                                 self.prob.g7, self.prob.g8, self.prob.g9, eq=eq, erg=erg, step_size=step, ineq=ineq, c=c)
        # problem.solve(max_iter=iter)
        # self.sol = problem.theta['z']

        # with open('trajs/temp/fly_optimized_trajectories.npy', 'wb') as f:
        #     np.save(f, np.array(self.sol[:, :2]))

        # self.files = glob.glob('optimized_trajectories.npy')

        with open('test_trajs/fix_control/noconverge_0.001_2.pkl', 'rb') as fp:
            self.traj = pickle.load(fp)
        # self.sol = np.load('optimized_trajectory.npy', allow_pickle=False)


        print("start")
        self.obs_subscription = self.create_subscription(Float32MultiArray, 'obs', self.get_obs_callback, 1)
        self.obs_subscription
        self.iter = 0
        self.sol = self.traj["x"]
        self.tf = self.traj["tf"]
        timer_period_sec = self.tf/len(self.sol)
        print(timer_period_sec)
        self.vid = True
        self.publisher_ = self.create_publisher(Float32MultiArray, 'action', 1)
        self.vid_publisher_ = self.create_publisher(Bool, 'vid', 1)
        self.land_publisher_ = self.create_publisher(Bool, 'land', 1)
        self.timer = self.create_timer(timer_period_sec, self.action_calculator)
        # self.vid_timer = self.create_timer(timer_period_sec, self.vid_calculator)
        self._pose  = Pose()
        self._twist = Twist()
        self._obs = []
        self.phis = 1.
        self.x_opt = []
        self.u_opt = []

        

        self.SIM_FREQ = freq
        self.TIMESTEP = 1./self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps


    def get_obs_callback(self, msg):
        self._obs = msg.data
        self._pose.position.x = msg.data[0]
        self._pose.position.y = msg.data[1]
        self._pose.position.z = msg.data[2]
        self._pose.orientation.x = msg.data[3]
        self._pose.orientation.y = msg.data[4]
        self._pose.orientation.z = msg.data[5]
        self._twist.linear.x = msg.data[10]
        self._twist.linear.y = msg.data[11]
        self._twist.linear.z = msg.data[12]
        self._twist.angular.x = msg.data[13]
        self._twist.angular.y = msg.data[14]
        self._twist.angular.z = msg.data[15]

    
    def action_calculator(self):
        if self.iter < 50:
            self.x_opt = self.sol[0]

            target = [float(self.x_opt[0]), 
                    float(self.x_opt[1]), 
                    float(self.x_opt[2])]

            msg = Float32MultiArray()
            msg.data = target
            self.publisher_.publish(msg)
        elif self.iter < 550:
            self.x_opt = self.sol[self.iter-50]

            target = [float(self.x_opt[0]), 
                    float(self.x_opt[1]), 
                    float(self.x_opt[2])]

            msg = Float32MultiArray()
            msg.data = target
            self.publisher_.publish(msg)
        else:
            self.x_opt = self.sol[-1]
            target = [float(self.x_opt[0]), 
                    float(self.x_opt[1]), 
                    float(0.)]

            msg = Bool()
            msg.data = True
            self.land_publisher_.publish(msg)
        # elif self.iter < 300:
        #     self.x_opt = self.sol[-1]
        #     target = [float(self.x_opt[0]), 
        #             float(self.x_opt[1]), 
        #             float(self.x_opt[2])]

        #     msg = Float32MultiArray()
        #     msg.data = target
        #     self.publisher_.publish(msg)
        # else:
        #     self.x_opt = self.sol[-1]
        #     target = [float(self.x_opt[0]), 
        #             float(self.x_opt[1]), 
        #             float(0.)]

        #     msg = Bool()
        #     msg.data = True
        #     self.land_publisher_.publish(msg)
        
        self.iter += 1



    def vid_calculator(self):
        msg = Bool()
        msg.data = self.vid
        self.vid_publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    comp = Computation()
    rclpy.spin(comp)


if __name__ == '__main__':
    main()