import numpy as np 
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap

from build_solver import build_erg_time_opt_solver

import pickle as pkl

args = {
    'N' : 600, 
    'x0' : np.array([0.5, 0.1]),
    'xf' : np.array([2.0, 3.2]),
    'erg_ub' : 0.01,
    'alpha' : 0.2,
    'wrksp_bnds' : np.array([[0.,100.],[0.,100.]])
}
solver, obs = build_erg_time_opt_solver(args)
solver.solve(max_iter=10000, eps=1e-7)
sol = solver.get_solution()



## <---- below draws the objects ---->
# for obs in traj_opt.obs:
#     _patch = obs.draw()
#     plt.gca().add_patch(_patch)

X, Y = np.meshgrid(*[np.linspace(wks[0],wks[1]) for wks in args['wrksp_bnds']])
pnts = np.vstack([X.ravel(), Y.ravel()]).T

_mixed_vals = np.inf * np.ones_like(X)
for ob in obs:
    _vals = np.array([ob.distance(pnt) for pnt in pnts]).reshape(X.shape)
    _mixed_vals = np.minimum(_vals, _mixed_vals)

    plt.contour(X, Y, _vals.reshape(X.shape), levels=[-0.01,0.,0.01])

plt.plot(sol['x'][:,0], sol['x'][:,1],'g.')
plt.plot(sol['x'][:,0], sol['x'][:,1])

sol.update({
    'x': np.array(sol['x']),
    'u' : np.array(sol['u']),
    'tf': np.array(sol['tf'])
})

_file = open('test_traj_max_erg_002.pkl', 'wb')
pkl.dump(sol, _file)

plt.show()