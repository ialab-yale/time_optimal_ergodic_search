import jax
from functools import partial
from jax import value_and_grad, grad, jacfwd, vmap, jit, hessian
from jax.flatten_util import ravel_pytree
import jax.numpy as np

class AugmentedLagrangian(object):
    def __init__(self, x0, loss, eq_constr, ineq_constr, args=None, step_size=1e-3, c=1.0):
        self.def_args    = args
        self.loss        = loss 
        self.c           = c
        self.eq_constr   = eq_constr
        self.ineq_constr = ineq_constr
        _eq_constr       = eq_constr(x0, args)
        _ineq_constr     = ineq_constr(x0, args)
        lam              = np.zeros(_eq_constr.shape)
        mu               = np.zeros(_ineq_constr.shape)
        self._x_shape    = x0.shape
        self.solution    = {'x' : x0, 'lam' : lam, 'mu' : mu}
        self.avg_sq_grad = np.zeros_like(x0)
        self._prev_val   = None
        # self._flat_solution, self._unravel = ravel_pytree(self.solution)

        def lagrangian(solution, args, c):
            # solution = self._unravel(flat_solution)
            x   = solution['x']
            lam = solution['lam']
            mu  = solution['mu']
            _eq_constr   = eq_constr(x, args)
            _ineq_constr = ineq_constr(x, args)
            return loss(x, args) \
                + np.sum(lam * _eq_constr + c*0.5 * (_eq_constr)**2) \
                + (1/c)*0.5 * np.sum(np.maximum(0., mu + c*_ineq_constr)**2 - mu**2)

        val_dldx = jit(value_and_grad(lagrangian))
        gamma   = 0.9
        eps     = 1e-8

        @jit
        def step(solution, args, avg_sq_grad, c):
            _val, _dldx   = val_dldx(solution, args, c)
            # _eps    = np.linalg.norm(_dldx['x'])
            avg_sq_grad = avg_sq_grad * gamma + np.square(_dldx['x']) * (1. - gamma)
            solution['x']   = solution['x'] - step_size * _dldx['x'] / np.sqrt(avg_sq_grad + eps)
            solution['lam'] = solution['lam'] + c*eq_constr(solution['x'], args)
            solution['mu']  = np.maximum(0, solution['mu'] + c*ineq_constr(solution['x'], args))
            return solution, _val, avg_sq_grad

        self.lagrangian      = lagrangian
        self.grad_lagrangian = val_dldx
        self.step = step

    def get_solution(self):
        return self.solution
        # return self._unravel(self._flat_solution)

    def set_init_cond(self, x):
        self.solution.update({'x' : x.copy()})

    def solve(self, args=None, max_iter=100000, eps=1e-7):
        if args is None:
            args = self.def_args
        _eps        = 1.0
        _prev_val   = None

        for k in range(max_iter):
            self.solution, _val, self.avg_sq_grad = self.step(self.solution, args, self.avg_sq_grad, self.c)
            # self.c = 1.001*self.c
            if _prev_val is None:
                _prev_val = _val
            else:
                # print(_val)
                _eps = np.abs(_val - _prev_val)
                _prev_val = _val
            if _eps < eps:
                print('done in ', k, ' iterations')
                return True

            # if k % 100 == 0:
            #     print('iter ', k, 'cost val ', self._prev_val, 'curr_tol ', _eps)
        return False


if __name__=='__main__':
    '''For testing purposes'''
    def f(x, args=None) : return 13*x[0]**2 + 10*x[0]*x[1] + 7*x[1]**2 + x[0] + x[1]
    def g(x, args) : return np.array([2*x[0]-5*x[1]-2])
    def h(x, args) : return x[0] + x[1] -1

    x0 = np.array([.5,-0.3])
    opt = AugmentedLagrangian(x0,f,g,h, step_size=0.1)
    opt.solve(max_iter=1000)
    sol = opt.get_solution()
    print(f(sol['x']), sol['x'])