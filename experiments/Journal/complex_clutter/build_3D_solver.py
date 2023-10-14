import sys 
sys.path.append('../../..')

import jax
from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian, value_and_grad
from jax.lax import scan
# from jax.ops import index_update, index
import jax.random as jnp_random
import jax.numpy as np

from jax.flatten_util import ravel_pytree

import numpy as onp
from time_opt_erg_lib.dynamics import DoubleIntegrator, SingleIntegrator, ThreeDAirCraftModel

from time_opt_erg_lib.ergodic_metric import ErgodicMetric
from time_opt_erg_lib.obstacle import Obstacle
from time_opt_erg_lib.cbf import constr2CBF
from time_opt_erg_lib.fourier_utils import BasisFunc, get_phik, get_ck
from time_opt_erg_lib.target_distribution import TargetDistribution
from time_opt_erg_lib.cbf_utils import sdf3cbfhole
from IPython.display import clear_output
import matplotlib.pyplot as plt

from time_opt_erg_lib.opt_solver import AugmentedLagrangeSolver
import yaml
import pickle as pkl

class TargetDistribution(object):
    def __init__(self) -> None:
        self.n = 3
        self.domain = np.meshgrid(
            *[np.linspace(0,1)]*self.n
        )
        self._s = np.stack([X.ravel() for X in self.domain]).T
        self.evals = (
            vmap(self.p)(self._s) , self._s
        )

    def p(self, x):
        return 1.0

def build_erg_time_opt_solver():
    basis           = BasisFunc(n_basis=[8,8,8])
    erg_metric      = ErgodicMetric(basis)
    robot_model     = ThreeDAirCraftModel()
    n,m = robot_model.n, robot_model.m
    target_distr    = TargetDistribution()

    args = {
        'N' : 500, 
        'x0' : np.array([2.5, 0.1, 2.5, 0., np.pi/2]),
        'xf' : np.array([2.5, 9.0, 2.5, 0., np.pi/2]),
        'erg_ub' : 0.0005,
        'alpha' : 0.2,
        'wrksp_bnds' : np.array([[0.,5],[0.,10],[0.,5.]])
    }
    _key = jnp_random.PRNGKey(0)

    _N_obs = 5
    obs = []
    cbf_constr = []

    for i in range(_N_obs):
        _key, _subkey = jnp_random.split(_key)
        _pos = jnp_random.uniform(_subkey, shape=(3,), minval=np.array([0.,0.,0.]), maxval=np.array([5.,10.,5.]))
        _radout = onp.array([0.5, 0.25, 0.5])
        _radin = onp.array([0.25, 0.25, 0.25])
        _rot = 0.

        _ob_inf_out = {
            'pos' : _pos, 
            'half_dims' : _radout,
            'rot': _rot
        }
        _ob_inf_in = {
            'pos' : _pos, 
            'half_dims' : _radin,
            'rot': _rot
        }
        _ob_out = Obstacle(_ob_inf_out)
        _ob_in = Obstacle(_ob_inf_in)
        obs.append(_ob_out)
        cbf_constr.append(sdf3cbfhole(robot_model.dfdt, _ob_out.distance3, _ob_in.distance3))

    # for i in range(_N_obs):
    #     _key, _subkey = jnp_random.split(_key)
    #     _pos = jnp_random.uniform(_subkey, shape=(3,), 
    #                     minval=np.array([0.,0.,0.]), maxval=np.array([5.,10.,5.]))
    #     _key, _subkey = jnp_random.split(_key)
    #     _rad = jnp_random.uniform(_subkey, shape=(3,), minval=0.5, maxval=.75) 
    #     _ob_inf = {
    #         'pos' : onp.array(_pos), 
    #         'half_dims' : onp.array(_rad),
    #         'rot': 0.
    #     }
    #     # _ob = Obstacle(_ob_inf, p=2)
    #     print(type(onp.array(_pos)))
    #     print(type(_rad))
    #     _ob = Obstacle(onp.array(_pos), onp.array(_rad), -np.pi*0./180., _ob_inf, p=2)
    #         # pos=np.array(obs_info[obs_name]['pos']), 
    #         # half_dims=np.array(obs_info[obs_name]['half_dims']),
    #         # th=obs_info[obs_name]['rot']
    #     obs.append(_ob)
    #     cbf_constr.append(sdf3cbf(robot_model.dfdt, _ob.distance3))


    args.update({
        'phik' : get_phik(target_distr.evals, basis),
    })

    ## <--- I DO NOT LIKE THIS
    workspace_bnds = args['wrksp_bnds']


    @vmap
    def emap(x):
        """ Function that maps states to workspace """
        return np.array([
            (x[0]-workspace_bnds[0][0])/(workspace_bnds[0][1]-workspace_bnds[0][0]), 
            (x[1]-workspace_bnds[1][0])/(workspace_bnds[1][1]-workspace_bnds[1][0]),
            (x[2]-workspace_bnds[2][0])/(workspace_bnds[2][1]-workspace_bnds[2][0])        
            ])
                
    def barrier_cost(e):
        """ Barrier function to avoid robot going out of workspace """
        return (np.maximum(0, e-1) + np.maximum(0, -e))**2

    # @jit
    def loss(params, args):
        x = params['x']
        u = params['u']
        tf = params['tf']
        N = args['N']
        dt = tf/N
        e = emap(x)
        """ Traj opt loss function, not the same as erg metric """
        return np.sum(barrier_cost(e)) + tf

    def eq_constr(params, args):
        """ dynamic equality constriants """
        x = params['x']
        u = params['u']

        x0 = args['x0']
        xf = args['xf']
        tf = params['tf']
        N  = args['N']
        dt = tf/N
        return np.vstack([
            x[0] - x0, 
            x[1:,:]-(x[:-1,:]+dt*vmap(robot_model.dfdt)(x[:-1,:], u[:-1,:])),
            x[-1] - xf
        ])

    def ineq_constr(params, args):
        """ inequality constraints"""
        x       = params['x']
        u       = params['u']
        phik    = args['phik']
        tf      = params['tf']
        N = args['N']
        dt = tf/N
        e = emap(x)
        _cbf_ineq = [vmap(_cbf_ineq, in_axes=(0,0,None, None))(x, u, args['alpha'], dt).flatten() 
                    for _cbf_ineq in cbf_constr]
        ck = get_ck(e, basis, tf, dt)
        _erg_ineq = [np.array([erg_metric(ck, phik) - args['erg_ub'], -tf])]
        _ctrl_box = [(-u[:,0]+.5).flatten(), (u[:,0]-5.0).flatten(), (np.abs(u[:,1:]) - np.pi/3).flatten()]
        return np.concatenate(_erg_ineq + _ctrl_box + _cbf_ineq)


    x = np.linspace(args['x0'], args['xf'], args['N'], endpoint=True)
    u = np.zeros((args['N'], robot_model.m))
    init_sol = {'x': x, 'u' : u, 'tf': np.array(30.0)}
    solver = AugmentedLagrangeSolver(
                    init_sol,
                    loss, 
                    eq_constr, 
                    ineq_constr, 
                    args, 
                    step_size=0.001,
                    c=0.6)
    return solver, obs, args