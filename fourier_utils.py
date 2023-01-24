from typing import List
import jax.numpy as np
from jax import vmap
from functools import partial
import numpy as onp

def get_hk(k): # normalizing factor for basis function
    _hk = (2. * k + onp.sin(2 * k))/(4. * k)
    _hk = _hk.at[onp.isnan(_hk)].set(1.)
    return onp.sqrt(onp.prod(_hk))

def get_ck(trajectory, basis, tf, dt):
    ck = np.sum(vmap(basis.fk_vmap)(trajectory), axis=0)
    ck = ck / basis.hk_list
    ck = ck * dt / tf
    return ck

def get_phik(vals, basis):
    _phi, _x = vals 
    phik = np.dot(_phi, vmap(basis.fk_vmap)(_x))
    phik = phik/phik[0]
    phik = phik/basis.hk_list
    return phik

def recon_from_fourier(basis_coef):
    pass

class BasisFunc(object):
    def __init__(self, n_basis) -> None:
        kmesh = np.meshgrid(
                    *[np.arange(0,n_max, step=1) for n_max in n_basis]
                )
        self.n = len(n_basis)
        self.k_list = np.stack([
                _k.ravel() for _k in kmesh
        ]).T * np.pi 

        self.hk_list = np.array([
            get_hk(_k) for _k in self.k_list
        ])

        self._fk = lambda k, x: np.prod(np.cos(x*k))
        self.fk_kvmap = vmap(self._fk, in_axes=(0, None))
        self.fk_xvmap = vmap(self._fk, in_axes=(None, 0))
        # self.fk_vmap = partial(self.fk_kvmap, np.array([0.1,0.2]))
        self.fk_vmap = partial(self.fk_kvmap, self.k_list)