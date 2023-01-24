import jax.numpy as np


class ErgodicMetric(object):

    def __init__(self, basis) -> None:
        self.basis = basis
        self.lamk = (1.+np.linalg.norm(basis.k_list/np.pi,axis=1)**2)**(-(basis.n+1)/2.)
        # lamk = np.exp(-0.8 * np.linalg.norm(k, axis=1))
        # lamk = np.ones((len(k), 1))
    def __call__(self, ck, phik):
        return np.sum(self.lamk * (ck - phik)**2)