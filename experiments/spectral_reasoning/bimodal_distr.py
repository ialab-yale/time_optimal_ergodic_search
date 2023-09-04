import jax.numpy as np
from jax import vmap

import matplotlib.pyplot as plt


class TargetDistribution(object):
    def __init__(self) -> None:
        self.n = 1
        self.domain = np.meshgrid(
            *[np.linspace(0,1,num=100)]*self.n
        )
        self._s = np.stack([X.ravel() for X in self.domain]).T
        _p = vmap(self.p)(self._s)
        _p = _p/np.mean(_p)
        self.evals = (
            _p , self._s
        )
    
    def plot(self):
        plt.contour(self.domain[0], self.domain[1], self.evals[0].reshape(self.domain[0].shape))

    def p(self, x):
        amp = 0.1
        return 0.8*np.exp(-180 * (x[0]-0.2)**2) + 0.7*np.exp(-400*(x[0]-0.68)**2) + 0.5*np.exp(-280*(x[0]-0.8)**2) + 0.8*np.exp(-700 * (x[0]-0.45)**2)

        # return np.sin(x[0]*2*np.pi*5+0.1)*np.exp(-180 * (x[0]-0.2)**2) + 0.7*np.exp(-180*(x[0]-0.68)**2) + 1.0
        # return np.exp(-150.5 * np.sum((x[:2] - 0.2)**2)) \
        #         + np.exp(-150.5 * np.sum((x[:2] - 0.75)**2)) \
        #         + np.exp(-150.5 * np.sum((x[:2] - np.array([0.2, 0.75]))**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - np.array([0.75, 0.2]))**2))
        # return 1
        # return np.exp(-60.5 * np.sum((x[:2] - 0.2)**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - 0.75)**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - np.array([0.2, 0.75]))**2)) \
        #             + np.exp(-60.5 * np.sum((x[:2] - np.array([0.75, 0.2]))**2))

    def update(self):
        pass