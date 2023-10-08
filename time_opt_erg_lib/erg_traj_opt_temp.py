### Jax/numpoy imports 
from time import time
import jax
import jax.numpy as np

from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian
from jax.lax import scan
import jax.random as jnp_random
from jax.flatten_util import ravel_pytree
import numpy as onp

### TODO
import pickle as pkl ## <--- this will probably get pushed up to user side

### Local imports
from .solver import AugmentedLagrangian
from .dynamics import SingleIntegrator
from .ergodic_metric import ErgodicMetric
from .obstacle import Obstacle
from .cbf_utils import sdf2cbf, sdf2cbfhole
from .fourier_utils import BasisFunc, get_phik, get_ck
from .target_distribution import TargetDistribution


class ErgodicTrajectoryOpt(object):
    def __init__(self, robot_model, obstacles, 
                        basis=None, time_horizon=500, args=None) -> None:
        self.time_horizon    = time_horizon
        self.robot_model     = robot_model
        if basis is None:
            self.basis = BasisFunc(n_basis=[8]*2)   # taking the first 8 modes
        else:
            self.basis = basis
        self.erg_metric      = ErgodicMetric(self.basis)
        n,m = self.robot_model.n, self.robot_model.m
        if args is None:
            self.target_distr = TargetDistribution()
            def_args = {
                'x0' : np.array([2.00,  3.25, 0.]),
                'xf' : np.array([1.75, -0.75, 0.]),
                'phik' : get_phik(self.target_distr.evals, self.basis),
                'wrksp_bnds' : np.array([[0.,3.5],[-1.,3.5]])
            }
            args = def_args
        self.def_args = args

        ### initial conditions 
        x = np.linspace(args['x0'], args['xf'], time_horizon, endpoint=True)
        u = np.zeros((time_horizon, self.robot_model.m))
        self.init_sol = np.concatenate([x, u], axis=1)
        self.curr_sol = (onp.array(x), onp.array(u))
        self.obs = obstacles

        # self.cbf_consts = []
        # for obs in self.obs: 
        #     self.cbf_consts.append(sdf2cbf(self.robot_model.f, obs.distance))

        self.cbf_consts_out = []
        self.cbf_consts_in = []
        for i in range(len(self.obs)):
            if i%2 == 0:
                self.cbf_consts_out.append(sdf2cbf(self.robot_model.f, self.obs[i].distance))
            else:
                self.cbf_consts_in.append(sdf2cbf(self.robot_model.f, self.obs[i].distance))
        # for obs in self.obs: 
        #     if len(temp) == 1:
        #         temp.append(obs)
        #         self.cbf_consts.append(sdf2cbfhole(self.robot_model.f, temp[0].distance, temp[1].distance))
        #         temp = []
        #     else:
        #         temp.append(obs)

        def _emap(x, args):
            """ Function that maps states to workspace """
            wrksp_bnds = args['wrksp_bnds']
            return np.array([
                (x[0]-wrksp_bnds[0,0])/(wrksp_bnds[0,1]-wrksp_bnds[0,0]), 
                (x[1]-wrksp_bnds[1,0])/(wrksp_bnds[1,1]-wrksp_bnds[1,0])])
        emap = vmap(_emap, in_axes=(0, None))
        
        def barrier_cost(e):
            """ Barrier function to avoid robot going out of workspace """
            return (np.maximum(0, e-1) + np.maximum(0, -e))**2
        
        @jit
        def loss(z, args):
            """ Traj opt loss function, not the same as erg metric """
            x, u = z[:, :n], z[:, n:]
            phik = args['phik']
            e = emap(x, args)
            ck = get_ck(e, self.basis, self.time_horizon, self.time_horizon/args['N'])
            return 5*self.erg_metric(ck, phik) \
                    + 0.1 * np.mean(u**2) \
                    + np.sum(barrier_cost(e))

        def eq_constr(z, args):
            """ dynamic equality constriants """
            x, u = z[:, :n], z[:, n:]
            x0 = args['x0']
            xf = args['xf']
            return np.vstack([
                x[0] - x0, 
                x[1:,:]-vmap(self.robot_model.f)(x[:-1,:], u[:-1,:]),
                x[-1] - xf
            ])

        def ineq_constr(z, args):
            """ control inequality constraints"""
            x, u = z[:, :n], z[:, n:]
            # p = x[:,:2] # extract just the position component of the trajectory
            # obs_val = [vmap(_ob.distance)(p).flatten() for _ob in self.obs]
            # obs_val = [vmap(_cbf_ineq, in_axes=(0,0,None))(x, u, args['alpha']).flatten() for _cbf_ineq in self.cbf_consts]
            obs_val = []
            for i in range(len(self.cbf_consts_out)):
                outs = vmap(self.cbf_consts_out[i], in_axes=(0,0,None))(x, u, args['alpha']).flatten()
                ins = vmap(self.cbf_consts_in[i], in_axes=(0,0,None))(x, u, args['alpha']).flatten()
                comb = list(map(max, zip(outs, -1*ins)))
                obs_val.append(comb)

            ctrl_box = [(np.abs(u) - 6.).flatten()]
            _ineq_list = ctrl_box + obs_val
            return np.concatenate(_ineq_list)

        self.eq_constr = eq_constr
        self.ineq_constr = ineq_constr

        self.solver = AugmentedLagrangian(
                                            self.init_sol,
                                            loss, 
                                            eq_constr, 
                                            ineq_constr, 
                                            args, 
                                            step_size=0.01,
                                            c=0.1
                    )
                        
        @jit
        def eval_erg_metric(x, args):
            """ evaluates the ergodic metric on a trjaectory  """
            x
            phik = args['phik']
            e    = emap(x, args)
            ck   = get_ck(e, self.basis)
            return self.erg_metric(ck, phik)
        self.eval_erg_metric = eval_erg_metric
        
    def get_trajectory(self, max_iter=10000, args=None):
        x = np.linspace(args['x0'], args['xf'], self.time_horizon, endpoint=True)
        u = np.zeros((self.time_horizon, self.robot_model.m))
        self.init_sol = np.concatenate([x, u], axis=1)
        self.solver.set_init_cond(self.init_sol)
        ifConv = self.solver.solve(max_iter=max_iter, args=args)
        sol = self.solver.get_solution()
        x = sol['x'][:,:self.robot_model.n]
        u = sol['x'][:,self.robot_model.n:]
        self.curr_sol = (x, u)
        return (x, u), ifConv

# <-- example code for how to use 
# if __name__=='__main__':
#     import sys 
#     import matplotlib.pyplot as plt

#     robot_model     = SingleIntegrator()
#     target_distr    = TargetDistribution()
#     basis           = BasisFunc(n_basis=[8,8])
#     args = {
#         'x0' : np.array([2.0,3.25, 0.]),
#         'xf' : np.array([1.75, -0.75, 0.]),
#         'phik' : get_phik(target_distr.evals, basis),
#         'wrksp_bnds' : np.array([[0.,3.5],[-1.,3.5]])
#     }
#     traj_opt = ErgodicTrajectoryOpt(robot_model, basis=basis, time_horizon=250, args=args)
#     print('solving traj')
#     x, u = traj_opt.get_trajectory()

#     # plotting function
#     for obs in traj_opt.obs:
#         _patch = obs.draw()
#         plt.gca().add_patch(_patch)
#     # _mixed_vals = np.inf*np.ones_like(X)
#     # for obs in traj_opt.obs:
#     #     _vals = vmap(obs.distance)(pnts).reshape(X.shape)
#     #     _mixed_vals = np.minimum(_vals, _mixed_vals)
#     #     plt.contour(X, Y, _vals.reshape(X.shape), levels=[-0.01,0.,0.01])
#     plt.plot(x[:,0], x[:,1], 'r')
#     plt.show()