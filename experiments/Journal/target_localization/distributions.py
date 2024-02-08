import jax.numpy as np
from jax import vmap
import numpy as onp

import matplotlib.pyplot as plt

import rospy 
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension

class ExpectedInformation(object):
    def __init__(self, wrksp_bnds) -> None:
        self.n = 2
        len_x = wrksp_bnds[0][1]-wrksp_bnds[0][0]
        len_y = wrksp_bnds[1][1]-wrksp_bnds[1][0]
        Nx = int(len_x*20)
        Ny = int(len_y*20)
        self.domain = np.meshgrid(
            *[
                np.linspace(wrksp_bnds[0][0],wrksp_bnds[0][1],num=Nx),
                np.linspace(wrksp_bnds[1][0],wrksp_bnds[1][1],num=Ny)
            ]
        )
        self._s = np.stack([X.ravel() for X in self.domain]).T
        self.evals = (
            vmap(self.p)(self._s) , self._s
        )

        self._target_dist_pub = rospy.Publisher('/target_dist', GridMap, queue_size=1)

        gridmap = GridMap()
        arr = Float32MultiArray()

        arr.data = onp.array(self.evals[0][::-1])
        arr.layout.dim.append(MultiArrayDimension())
        arr.layout.dim.append(MultiArrayDimension())

        arr.layout.dim[0].label="column_index"
        arr.layout.dim[0].size=Ny
        arr.layout.dim[0].stride=Ny*Ny

        arr.layout.dim[1].label="row_index"
        arr.layout.dim[1].size=Nx
        arr.layout.dim[1].stride=Nx


        gridmap.layers.append("elevation")
        gridmap.data.append(arr)
        gridmap.info.length_x=wrksp_bnds[0][1]-wrksp_bnds[0][0]
        gridmap.info.length_y=wrksp_bnds[1][1]-wrksp_bnds[1][0]
        gridmap.info.pose.position.x=onp.mean(wrksp_bnds[0,:])
        gridmap.info.pose.position.y=onp.mean(wrksp_bnds[1,:])

        gridmap.info.header.frame_id = "world"
        gridmap.info.resolution = 0.05

        self._grid_msg = gridmap
    
    def pub_map(self):
        self._target_dist_pub.publish(self._grid_msg)

    def plot(self):
        plt.contour(self.domain[0], self.domain[1], self.evals[0].reshape(self.domain[0].shape))

    def p(self, x):
        # return 0.25*(np.exp(-10.5 * np.sum((x[:2] - np.array([1.0, -0.5]))**2)) \
        #         + np.exp(-10.5 * np.sum((x[:2] - np.array([2.5, .0]))**2)) \
        #         + np.exp(-10.5 * np.sum((x[:2] - np.array([1.2, 2.0]))**2)) \
        #             + np.exp(-10.5 * np.sum((x[:2] - np.array([2.5, 3.0]))**2)))
        return 0.001
        # return np.exp(-60.5 * np.sum((x[:2] - 0.2)**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - 0.75)**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - np.array([0.2, 0.75]))**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - np.array([0.75, 0.2]))**2))
        # return 0.25*(np.exp(-10.5 * np.sum((x[:1] - np.array([1.75]))**2)))
        # return 0.5*(np.exp(-10.5 * np.sum((x[:2] - np.array([0.5, 1.5]))**2)) \
        #         + np.exp(-10.5 * np.sum((x[:2] - np.array([3.0, 1.5]))**2)))
        
    def run(self):
        while not rospy.is_shutdown():
            self._target_dist_pub.publish(self._grid_msg)
            self._rate.sleep()



class TargetBelief(object):
    def __init__(self, wrksp_bnds) -> None:
        self.n = 2
        len_x = wrksp_bnds[0][1]-wrksp_bnds[0][0]
        len_y = wrksp_bnds[1][1]-wrksp_bnds[1][0]
        Nx = int(len_x*20)
        Ny = int(len_y*20)
        self.domain = np.meshgrid(
            *[
                np.linspace(wrksp_bnds[0][0],wrksp_bnds[0][1],num=Nx),
                np.linspace(wrksp_bnds[1][0],wrksp_bnds[1][1],num=Ny)
            ]
        )
        self._s = np.stack([X.ravel() for X in self.domain]).T
        self.evals = (
            vmap(self.p)(self._s) , self._s
        )

        self._target_belief_pub = rospy.Publisher('/target_belief', GridMap, queue_size=1)

        gridmap = GridMap()
        arr = Float32MultiArray()

        arr.data = onp.array(self.evals[0][::-1])
        arr.layout.dim.append(MultiArrayDimension())
        arr.layout.dim.append(MultiArrayDimension())

        arr.layout.dim[0].label="column_index"
        arr.layout.dim[0].size=Ny
        arr.layout.dim[0].stride=Ny*Ny

        arr.layout.dim[1].label="row_index"
        arr.layout.dim[1].size=Nx
        arr.layout.dim[1].stride=Nx


        gridmap.layers.append("elevation")
        gridmap.data.append(arr)
        gridmap.info.length_x=wrksp_bnds[0][1]-wrksp_bnds[0][0]
        gridmap.info.length_y=wrksp_bnds[1][1]-wrksp_bnds[1][0]
        gridmap.info.pose.position.x=onp.mean(wrksp_bnds[0,:])
        gridmap.info.pose.position.y=onp.mean(wrksp_bnds[1,:])

        gridmap.info.header.frame_id = "world"
        gridmap.info.resolution = 0.05

        self._grid_msg = gridmap
    
    def pub_map(self):
        self._target_belief_pub.publish(self._grid_msg)

    def plot(self):
        plt.contour(self.domain[0], self.domain[1], self.evals[0].reshape(self.domain[0].shape))

    def p(self, x):
        return 0.5*(np.exp(-10.5 * np.sum((x[:2] - np.array([0.5, 0.5]))**2)))
        
    def run(self):
        while not rospy.is_shutdown():
            self._target_belief_pub.publish(self._grid_msg)
            self._rate.sleep()


if __name__=="__main__":
    rospy.init_node('test_grid_map')
    wrksp_bnds = np.array([[0.,3.5],[-1.,3.5]])
    target_distr = ExpectedInformation(wrksp_bnds)
    # target_distr.plot()
    # plt.show()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        target_distr.pub_map()
        rate.sleep()
