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
from sensor_model import GaussianSensorModel, get_sensor_ck
from time_opt_erg_lib.dynamics import DoubleIntegrator, SingleIntegrator3D

from time_opt_erg_lib.ergodic_metric import ErgodicMetric
from time_opt_erg_lib.obstacle import Obstacle
from time_opt_erg_lib.cbf import constr2CBF
from time_opt_erg_lib.fourier_utils import BasisFunc, get_phik, get_ck
# from time_opt_erg_lib.target_distribution import TargetDistribution
from time_opt_erg_lib.cbf_utils import sdf2cbf
from IPython.display import clear_output
import matplotlib.pyplot as plt

from time_opt_erg_lib.opt_solver import AugmentedLagrangeSolver
import yaml
import pickle as pkl


def build_erg_time_opt_solver(args, target_distr):
    
    ## <--- I DO NOT LIKE THIS
    workspace_bnds = args['wrksp_bnds']

    # @vmap
    def emap(x):
        """ Function that maps states to workspace """
        return np.array([
            (x[0]-workspace_bnds[0][0])/(workspace_bnds[0][1]-workspace_bnds[0][0]), 
            (x[1]-workspace_bnds[1][0])/(workspace_bnds[1][1]-workspace_bnds[1][0])])
            
    
    basis           = BasisFunc(n_basis=[8,8], emap=emap)
    erg_metric      = ErgodicMetric(basis)
    robot_model     = SingleIntegrator3D()
    sensor          = GaussianSensorModel(np.array([0.1, 0.1]), np.array([0,1]))
    n,m = robot_model.n, robot_model.m

    
    args.update({
        'phik' : get_phik(target_distr.evals, basis),
    })


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
        e = vmap(emap)(x)
        """ Traj opt loss function, not the same as erg metric """
        return 75*np.sum(barrier_cost(e)) + tf

    def eq_constr(params, args):
        """ dynamic equality constriants """
        x = params['x']
        u = params['u']

        x0 = args['x0']
        xf = args['xf']
        tf = params['tf']
        N = args['N']
        dt = tf/N
        return np.vstack([
            x[0] - x0, 
            x[1:,:]-(x[:-1,:]+dt*vmap(robot_model.dfdt)(x[:-1,:], u[:-1,:])),
            x[-1] - xf
        ])

    def ineq_constr(params, args):
        """ inequality constraints"""
        x = params['x']
        u = params['u']
        phik = args['phik']
        tf = params['tf']
        N = args['N']
        dt = tf/N
        ck = get_ck(x, basis, tf, dt)
        ck = get_sensor_ck(x,sensor,target_distr,basis,N*dt,dt)
        erg = erg_metric(ck, phik)
        deb.print("erg: {a}", a=erg)
        _erg_ineq = [10*np.array([erg - args['erg_ub'], -tf])]
        _ctrl_box = [(np.abs(u) - 2.).flatten()]
        _ctrl_disk = [(u[:,0]**2 + u[:,1]**2 - 1.2345).flatten()]
        return np.concatenate(_erg_ineq + _ctrl_disk)



    x = np.linspace(args['x0'], args['xf'], args['N'], endpoint=True)
    u = np.zeros((args['N'], robot_model.m))
    init_sol = {'x': x, 'u' : u, 'tf': np.array(10.0)}
    solver = AugmentedLagrangeSolver(
                    init_sol,
                    loss, 
                    eq_constr, 
                    ineq_constr, 
                    args, 
                    step_size=1e-3,
                    c=1.0)
    return solver