import jax.numpy as np
from jax import vmap
from jax import hessian
import numpy as onp
import matplotlib.pyplot as plt


class BayesFilter(object):

    def __init__(self, 
                 meas_model, 
                 prior, 
                 wrksp_bnds) -> None:
        len_x = wrksp_bnds[0][1]-wrksp_bnds[0][0]
        len_y = wrksp_bnds[1][1]-wrksp_bnds[1][0]
        Nx = int(len_x*20)
        Ny = int(len_y*20)
        self.domain = np.meshgrid(
            *[
                np.linspace(wrksp_bnds[0][0],wrksp_bnds[0][1],num=50),
                np.linspace(wrksp_bnds[1][0],wrksp_bnds[1][1],num=50)
            ]
        )
        self.meas_model = meas_model
        self.score = hessian(meas_model)
        self._s = np.stack([X.ravel() for X in self.domain]).T
        self._prior = vmap(prior)(self._s)
        # normalize just in case 
        self._prior = self._prior/np.sum(self._prior)
        self.evals = (self._prior, self._s)
    def eid(self, x):
        _instant_eid = vmap(self.score, in_axes=(0, None))(self._s, x)
        return np.sum(vmap(np.dot)(_instant_eid, self._prior), axis=0)
        
    def plot_eid(self):
        plt.contour(self.domain[0], self.domain[1], self.evals[0].reshape(self.domain[0].shape))

    def plot_prior(self):
        plt.contour(self.domain[0], self.domain[1], self._prior.reshape(self.domain[0].shape))
    def update_prior(self, x, y):
        self._prior = self._prior * np.exp(-10*(vmap(self.meas_model, in_axes=(0, None))(self._s, x)-y)**2)
        self._prior = self._prior + 1e-5
        self._prior = self._prior/np.sum(self._prior)
    def update_eid(self):   
        fish_val = lambda x: np.linalg.det(self.eid(x))
        self.evals = (vmap(fish_val)(self._s), self._s)

