import numpy as np 
import matplotlib.pyplot as plt

from build_solver import build_erg_time_opt_solver

import pickle as pkl

erg_ubs = [0.001, 0.005, 0.01, 0.05, 0.1]
for i, erg_ub in enumerate(erg_ubs):
    args = {
        'N' : 200, 
        'x0' : np.array([0.1, 0.1, 0., 0.]),
        'xf' : np.array([0.9, 0.9, 0., 0.]),
        'erg_ub' : erg_ub,
        # 'alpha' : 0.8,
    }
    solver = build_erg_time_opt_solver(args)
    solver.solve(max_iter=10000, eps=1e-7)
    sol = solver.get_solution()
    print(sol['tf'], erg_ub)
    plt.figure(i)
    plt.plot(sol['x'][:,0], sol['x'][:,1],'g.')
    plt.plot(sol['x'][:,0], sol['x'][:,1])
    

plt.show()