import jax.numpy as np
from jax import vmap
from jax import hessian, jacfwd
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
        def logp(p, x, y):
            return -10.5 * np.sum((meas_model(p, x) - y)**2)
        self.logp = logp
        self.score = jacfwd(meas_model)
        self._s = np.stack([X.ravel() for X in self.domain]).T
        self._prior = vmap(prior)(self._s)
        # normalize just in case 
        self._prior = self._prior/np.sum(self._prior)
        self.evals = (self._prior, self._s)
        self.fish_evals = (self._prior.copy(), self._s)
    def eid(self, x):
        _instant_eid = vmap(self.score, in_axes=(0, None))(self._s, x)
        _instant_eid = vmap(np.outer)(_instant_eid, _instant_eid)
        return np.sum(vmap(np.dot)(_instant_eid, self._prior), axis=0)
        
    def plot_eid(self):
        # plt.contourf(self.domain[0], self.domain[1], self.evals[0].reshape(self.domain[0].shape))
        plt.imshow(self.fish_evals[0].reshape(self.domain[0].shape), extent=(-2,2,-2,2), origin='lower')

    def plot_prior(self):
        # plt.imshow(self.domain[0], self.domain[1], self._prior.reshape(self.domain[0].shape))
        plt.imshow(self._prior.reshape(self.domain[0].shape), extent=(-2,2,-2,2), origin='lower')

    def update_prior(self, x, y):
        self._prior = self._prior * np.exp(vmap(self.logp,in_axes=(0, None,None))(self._s, x, y))
        self._prior = self._prior + 1e-5
        self._prior = self._prior/np.sum(self._prior)

    def update_eid(self):   
        fish_val = lambda x: np.linalg.det(self.eid(x))
        self.fish_evals = (vmap(fish_val)(self._s), self._s)
        return self.fish_evals
        # self.evals = (vmap(fish_val)(self._s), self._s)

