import sys
sys.path.append('../../../')
import numpy as np

from time_opt_erg_lib.dynamics import SingleIntegrator
from time_opt_erg_lib.target_distribution import TargetDistribution
from time_opt_erg_lib.fourier_utils import BasisFunc, get_phik
from time_opt_erg_lib.erg_traj_opt import ErgodicTrajectoryOpt
from time_opt_erg_lib.obstacle import Obstacle

import pickle as pkl
import matplotlib.pyplot as plt


if __name__=='__main__':

    robot_model     = SingleIntegrator()
    target_distr    = TargetDistribution()
    basis           = BasisFunc(n_basis=[8,8])
    wksp_bnds       = np.array([[0.,3.5],[-1.,3.5]])

    args = {
        'x0' : np.array([.1,.0]),
        'xf' : np.array([2., 3.2]),
        'phik' : get_phik(target_distr.evals, basis),
        'wrksp_bnds' : wksp_bnds,
        'alpha' : 0.2
    }

    obs_info = pkl.load(open('obs_info.pkl', 'rb'))
    obs = []
    for obs_name in obs_info:
        _ob = Obstacle(
            pos=np.array(obs_info[obs_name]['pos']), 
            half_dims=np.array(obs_info[obs_name]['half_dims']),
            th=obs_info[obs_name]['rot']
        )
        obs.append(_ob)
    
    def isSafe(x, obs):
        safe = True
        for ob in obs:
            if ob.distance(x) > 0:
                safe = False
                break
        return safe
            

    traj_opt = ErgodicTrajectoryOpt(robot_model, obstacles=obs, basis=basis, time_horizon=200, args=args)

    X, Y = np.meshgrid(*[np.linspace(wks[0],wks[1]) for wks in args['wrksp_bnds']])
    pnts = np.vstack([X.ravel(), Y.ravel()]).T

    _mixed_vals = -np.inf * np.ones_like(X)
    for ob in obs:
        _vals = np.array([ob.distance(pnt) for pnt in pnts]).reshape(X.shape)
        _mixed_vals = np.maximum(_vals, _mixed_vals)
    
    plt.figure(figsize=(3,2))

    # set the seed 
    np.random.seed(10)
    max_N    = 10
    succ_cnt = 0
    min_dist = 0.5
    while succ_cnt < max_N:
        x0 = np.random.uniform(wksp_bnds[:,0], wksp_bnds[:,1]) 
        if isSafe(x0, obs):
            dist = 0.
            while dist < min_dist:
                xf = np.random.uniform(wksp_bnds[:,0], wksp_bnds[:,1])
                if isSafe(xf, obs):
                    dist = np.linalg.norm(x0-xf)

            print('found candidate pair')
            print(x0, xf)
            args.update({'x0' : x0})
            args.update({'xf' : xf})
            print('solving traj')
            (x, u), ifConv = traj_opt.get_trajectory(args=args)
            if ifConv:
                print('solver converged')
                plt.contour(X, Y, _mixed_vals, levels=[-0.01,0.,0.01], linewidths=2, colors='k')
                plt.plot(x[:,0], x[:,1], linestyle='dashdot')#, c='m', alpha=alpha)

                plt.tight_layout()
                plt.axis('equal')

                plt.show()
                succ_cnt += 1
        # TODO need to add in data saving here 