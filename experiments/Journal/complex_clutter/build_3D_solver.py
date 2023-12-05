import sys 
sys.path.append('../../..')

import jax
from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian, value_and_grad
from jax.lax import scan
# from jax.ops import index_update, index
import jax.random as jnp_random
import jax.numpy as np
import jax.debug as deb

from jax.flatten_util import ravel_pytree

import numpy as onp
from time_opt_erg_lib.dynamics import SingleIntegrator2D, DoubleIntegrator, NDDoubleIntegrator, SingleIntegrator3D, ThreeDAirCraftModel, DroneDynamics

from time_opt_erg_lib.ergodic_metric import ErgodicMetric
from time_opt_erg_lib.obstacle import Obstacle, Torus
from time_opt_erg_lib.cbf import constr2CBF
from time_opt_erg_lib.fourier_utils import BasisFunc, get_phik, get_ck
from time_opt_erg_lib.target_distribution import TargetDistribution
from time_opt_erg_lib.cbf_utils import sdf3cbfhole, sdf3cbf
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
    robot_model     = SingleIntegrator3D()
    # robot_model     = ThreeDAirCraftModel()
    # robot_model     = DoubleIntegrator(dim=3)
    # robot_model     = DroneDynamics()
    n,m = robot_model.n, robot_model.m
    target_distr    = TargetDistribution()

    args = {
        'N' : 500, 
        'alpha': 1.00001,
        'erg_ub' : 0.001,
        # 'x0' : np.array([4., 0.1, 2.5, 0., np.pi/2]),   # Airplane
        # 'xf' : np.array([4., 9.0, 2.5, 0., np.pi/2]),   # Airplane
        # 'x0' : np.array([1., 0.1, 2.5, 0., 0., 0.]),    # Pointmass
        # 'xf' : np.array([8., 9.0, 2.5, 0., 0., 0.]),    # Pointmass
        'x0' : np.array([0.2, 0.2, .5]),    # SingleInt
        'xf' : np.array([3.2, 3.2, .5]),    # SingleInt
        # 'x0' : np.concatenate([np.array([1., 0.1, 2.5]), np.eye(3).ravel(), np.zeros(3), np.zeros(3)]),   # Drone
        # 'xf' : np.concatenate([np.array([8., 9., 2.5]), np.eye(3).ravel(), np.zeros(3), np.zeros(3)]),   # Drone
        # 'x0' : np.array([0., 0., .5, 0., np.pi/2]),    # Physical
        # 'xf' : np.array([3.5, 3.5, .5, 0., np.pi/2]),    # Physical
        # 'x0' : np.concatenate([np.array([0.2, 0.2, .5]), np.eye(3).ravel(), np.zeros(3), np.zeros(3)]),   # Physical
        # 'xf' : np.concatenate([np.array([3.2, 3.2, .5]), np.eye(3).ravel(), np.zeros(3), np.zeros(3)]),   # Physical
        # 'wrksp_bnds' : np.array([[0.,10.],[0.,10],[0.,7.]])   # Simulation
        'wrksp_bnds' : np.array([[0.1,3.4],[0.1,3.4],[0.2,1.4]])    # Physical
    }

    obs = []
    cbf_constr = []
    a = onp.load('obs_physical.npy', allow_pickle=True)
    for ele in a:
        t = Torus(ele)
        obs.append(t)
        # cbf_constr.append(sdf3cbf(robot_model.dfdt, t.distance))
        cbf_constr.append(sdf3cbf(robot_model.dfdt, t.distance))


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
        # deb.print("e: {a}", a=np.any((e<0)|(e>1)))
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
            # x[1:,:]-(x[:-1,:]+dt*vmap(robot_model.dfdt)(x[:-1,:], u[:-1,:])),
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
        # _sdf_ineq = 10*[-vmap(t.distance)(x[:,:3]) for t in obs]
        # deb.print("sdf: {a}", a=np.any(_sdf_ineq[0]>0))
        ck = get_ck(e, basis, tf, dt)
        _erg_ineq = [np.array([erg_metric(ck, phik) - args['erg_ub'], -tf])]
        # _ctrl_box = [(-u[:,0]+.5).flatten(), (u[:,0]-5.0).flatten(), (np.abs(u[:,1:]) - np.pi/3).flatten()]
        _ctrl_box = [(abs(u[:,:])-8.).flatten()]  # For single integrator dynamics
        return np.concatenate(_erg_ineq + _ctrl_box + _cbf_ineq)
        # return np.concatenate(_erg_ineq + _cbf_ineq)
        # return np.concatenate(_erg_ineq + _ctrl_box )


    x = np.linspace(args['x0'], args['xf'], args['N'], endpoint=True)
    u = np.zeros((args['N'], robot_model.m))
    init_sol = {'x': x, 'u' : u, 'tf': np.array(22.0)}
    solver = AugmentedLagrangeSolver(
                    init_sol,
                    loss, 
                    eq_constr, 
                    ineq_constr, 
                    args, 
                    step_size=0.001,
                    c=1.0)
    return solver, obs, args