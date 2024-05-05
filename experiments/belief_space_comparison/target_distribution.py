import jax.numpy as np
from jax import vmap
import numpy as onp

import matplotlib.pyplot as plt


class TargetDistribution(object):
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
    
    def pub_map(self):
        self._target_dist_pub.publish(self._grid_msg)

    def plot(self):
        plt.contour(self.domain[0], self.domain[1], self.evals[0].reshape(self.domain[0].shape))

    def p(self, x):
        return 1/(1 + )
    0.25*(np.exp(-10.5 * np.sum((x[:2] - np.array([1.0, -0.5]))**2)) \
                + np.exp(-10.5 * np.sum((x[:2] - np.array([2.5, .0]))**2)) \
                + np.exp(-10.5 * np.sum((x[:2] - np.array([1.2, 2.0]))**2)) \
                    + np.exp(-10.5 * np.sum((x[:2] - np.array([2.5, 3.0]))**2)))
