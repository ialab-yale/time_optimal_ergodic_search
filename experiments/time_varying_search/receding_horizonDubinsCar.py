import sys 
sys.path.append('../..')

import jax
from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian, value_and_grad
from jax.lax import scan
# from jax.ops import index_update, index
import jax.random as jnp_random
import jax.numpy as np

from jax.flatten_util import ravel_pytree

import numpy as onp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', type=int, default=69)
parser.add_argument('-th', '--timehorizon', type=int, default=10)
parser.add_argument('-trial', '--trial', type=int, default=0)
parsed_args = parser.parse_args()

onp.random.seed(parsed_args.seed)

from time_opt_erg_lib.dynamics import DoubleIntegrator, KinematicUnicycle

from time_opt_erg_lib.ergodic_metric import ErgodicMetric
from time_opt_erg_lib.obstacle import Obstacle
from time_opt_erg_lib.cbf import constr2CBF
from time_opt_erg_lib.fourier_utils import BasisFunc, get_phik, get_ck, recon_from_fourier
# from time_opt_erg_lib.target_distribution import TargetDistribution
from time_opt_erg_lib.cbf_utils import sdf2cbf
from IPython.display import clear_output
import matplotlib.pyplot as plt

from opt_solver import AugmentedLagrangeSolver
import yaml
import dill as pkl

from bayes_filter import BayesFilter

class Logger:
    def __init__(self, true_loc):
        self.true_loc = true_loc
        self.means      = []
        self.stds       = []
        self.infs        = []
        self.probs       = []
        self.sample_locs = []
        self.plans      = []
        self.time       = []
        self.dts        = []

    def append(self, mean, std, inf, prob, sample_loc, plan, t_curr, dt):
        self.means.append(mean)
        self.stds.append(std)
        self.infs.append(onp.array(inf))
        self.probs.append(onp.array(prob))
        self.sample_locs.append(sample_loc)
        self.plans.append(plan)
        self.time.append(t_curr)
        self.dts.append(dt)

    def to_numpy(self):
        self.means = onp.vstack(self.means)
        self.sample_locs = onp.vstack(self.sample_locs)
        self.time = onp.array(self.time)

class CkDynamics(object):
    def __init__(self, basis, _bnds) -> None:
        self._bnds = _bnds
        self.basis = basis
        self._ck_state = np.zeros(basis.k_list.shape[0])
        self._t_curr = 0.0
        self.domain = np.meshgrid(
            *[
                np.linspace(_bnds[0][0],_bnds[0][1]),
                np.linspace(_bnds[1][0],_bnds[1][1])
            ]
        )
        self.x_vals = np.stack([X.ravel() for X in self.domain]).T
    def recon_ck(self):
        return recon_from_fourier(self._ck_state/self._t_curr, self.basis, self.basis.k_list, self.x_vals)
    def plot_ck(self):
        evals = self.recon_ck()
        plt.contour(self.domain[0], self.domain[1], evals.reshape(self.domain[0].shape))
    def step(self, x, dt):
        self._ck_state = self._ck_state + dt * self.basis.fk_vmap(x)
        self._t_curr = self._t_curr + dt
    
class ErgodicPlanner(object):
    def __init__(self, target_distribution, args) -> None:
        _bnds = args['wrksp_bnds']
        def emap(x):
            """ Function that maps states to workspace """
            return np.array([
                (x[0]-_bnds[0][0])/(_bnds[0][1]-_bnds[0][0]), 
                (x[1]-_bnds[1][0])/(_bnds[1][1]-_bnds[1][0])])
        vmap_emap = vmap(emap)

        basis           = BasisFunc(n_basis=[10]*2, emap=emap)
        self._ck_dynamics = CkDynamics(basis, _bnds)
        self.basis = basis 
        erg_metric      = ErgodicMetric(basis)
        self.erg_metric = erg_metric
        robot_model     = KinematicUnicycle()
        n,m = robot_model.n, robot_model.m
        args.update({
            'phik' : get_phik(target_distribution.evals, basis),
            'ck_state' : self._ck_dynamics._ck_state
        })
        # @jit
        def loss(params, args):
            x = params['x']
            u = params['u']
            phik    = args['phik']
            tf      = args['tf']
            N       = args['N']
            dt      = tf/N
            ck = (get_ck(x, basis, tf, dt) * tf + args['ck_state'])/(args['t_curr'] + tf)            
            return 10*erg_metric(ck, phik) 

        def eq_constr(params, args):
            """ dynamic equality constriants """
            x = params['x']
            u = params['u']
            x0 = args['x0']
            xf = args['xf']
            tf = args['tf']
            N  = args['N']
            dt = tf/N
            return np.vstack([
                x[0] - x0, 
                x[1:,:]-(x[:-1,:]+dt*vmap(robot_model.dfdt)(x[:-1,:], u[:-1,:])),
                (x[-1] - xf)*0
            ])

        def ineq_constr(params, args):
            """ inequality constraints"""
            x = params['x']
            u = params['u']
            phik    = args['phik']
            tf      = args['tf']
            N       = args['N']
            e = vmap_emap(x)
            _ctrl_box = [(-u[:,0]+0.1).flatten(), (u[:,0] - 1.).flatten(), (np.abs(u[:,1]) - np.pi).flatten()]
            _expl_box = [(-e).flatten(), (e-1.0).flatten()]
            return np.concatenate( _ctrl_box + _expl_box)


        x = np.linspace(args['x0'], args['xf'], args['N'], endpoint=True)
        u = np.zeros((args['N'], robot_model.m))
        init_sol = {'x': x, 'u' : u}
        self.solver = AugmentedLagrangeSolver(
                        init_sol,
                        loss, 
                        eq_constr, 
                        ineq_constr, 
                        args, 
                        step_size=1e-3,
                        c=10.)
    def update_plan(self, args, max_iter=10_000):
        # self.solver.solve(args, max_iter=max_iter, eps=1e-8)
        self.solver.solve(args, max_iter=max_iter, eps=0.4, alpha=1.0001)
        return self.solver.get_solution()

    def time_shift_plan(self):
        self.solver.solution.update({
            'x' : self.solver.solution['x'].at[:-1].set(self.solver.solution['x'][1:]),
            'u' : self.solver.solution['u'].at[:-1].set(self.solver.solution['u'][1:])
        })

args = {
    'N' : 25, 
    'x0' : np.array([-0.9, -0.9, 0.]),
    'xf' : np.array([.9, .9, 0.]),
    'erg_ub' : 0.1,
    'wrksp_bnds' : np.array([[-2.,2.],[-2.,2.]]), 
    't_curr' : 0, 
    'tf' : float(parsed_args.timehorizon)
}

def meas_model(p, x):
    # return np.linalg.norm(x-p + 1e-3)
    # return np.arctan2(p[1]-x[1], (p[0]-x[0] + 1e-3))
    return 1.0/(1+np.sum((x-p)**2))

target_distribution = BayesFilter(meas_model, lambda x: 1.0, args['wrksp_bnds'])
robot = KinematicUnicycle()
planner = ErgodicPlanner(target_distribution, args)

_distractor_p = np.array([0.,0.])
# _true_p = onp.random.uniform(-2,2, size=(2,))
_true_p = np.array([.75, 0.75])
_robot_state = np.array(args['x0'])
dt_list = []

logger = Logger(_true_p)


t_lap = 0
_percent_max = 0.01
unif_phik = get_phik(target_distribution.evals, planner.basis)
args.update({'erg_ub' : _percent_max*planner.erg_metric(unif_phik, 2*unif_phik)})


t_max = 20 
while planner._ck_dynamics._t_curr < t_max:
    planner.solver.reset()
    sol = planner.update_plan(args)
    _dt = args['tf']/args['N']
    (mean, var) = target_distribution.get_mean_var()

    logger.append(mean, var, 
                  (target_distribution.fish_evals[0].copy()).reshape((50,50)), 
                  (target_distribution._prior.copy()).reshape((50,50)), 
                  _robot_state[:2].copy(), 
                  sol['x'].copy(), 
                  np.copy(planner._ck_dynamics._t_curr), 
                  _dt)
    
    t_lap = t_lap + _dt
    dt_list.append(_dt)
    planner._ck_dynamics.step(_robot_state, _dt)
    _robot_state = _robot_state + _dt * robot.dfdt(_robot_state, sol['u'][0])
    args.update({
        'x0': _robot_state,
        'ck_state' : planner._ck_dynamics._ck_state, 
        't_curr' : planner._ck_dynamics._t_curr
    })
    _y = meas_model(_true_p, _robot_state[:2]) + onp.random.normal(0., 0.1)
    # _y_distract = meas_model(_robot_state[:2], _distractor_p) + onp.random.uniform(0., 0.1)
    target_distribution.update_prior(_robot_state[:2], _y)
    # target_distribution.update_prior(_robot_state[:2], _y_distract)
    _fish_evals = target_distribution.update_eid()
    args.update({
        'phik' : get_phik(_fish_evals, planner.basis)
    })
    # args.update({'erg_ub' : _percent_max*(
    #     0*planner.erg_metric(unif_phik, args['phik']) + planner.erg_metric(0*unif_phik, args['phik'])
    #     )})
    args.update({'erg_ub' : _percent_max*planner.erg_metric(0*unif_phik, args['phik'])
    })
    t_lap = 0
    planner.time_shift_plan()
    if np.linalg.det(var) < 1e-2:
        print('time elapsed ', planner._ck_dynamics._t_curr)
        break
pkl.dump(logger, open('./data/receding_horizon_5_trial{}.pkl'.format(parsed_args.trial), 'wb'))