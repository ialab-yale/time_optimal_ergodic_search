import jax
from functools import partial
from jax import value_and_grad, grad, jacfwd, vmap, jit, hessian
from jax.flatten_util import ravel_pytree
import jax.numpy as np


class AugmentedLagrangeSolver(object):
    def __init__(self, x0, loss, eq_constr, ineq_constr, args=None, step_size=1e-3, c=1.0):
        self.def_args = args
        self.loss = loss 
        self.c = c
        self.eq_constr   = eq_constr
        self.ineq_constr = ineq_constr
        _eq_constr       = eq_constr(x0, args)
        _ineq_constr     = ineq_constr(x0, args)
        lam = np.zeros(_eq_constr.shape)
        mu  = np.zeros(_ineq_constr.shape)
        self.solution = x0
        self.dual_solution = {'lam' : lam, 'mu' : mu}
        self.avg_sq_grad = {}
        for _key in self.solution:
            self.avg_sq_grad.update({_key : np.zeros_like(self.solution[_key])})

        self._prev_val = None
        # self._flat_solution, self._unravel = ravel_pytree(self.solution)
        def lagrangian(solution, dual_solution, args, c):
            # solution = self._unravel(flat_solution)
            lam = dual_solution['lam']
            mu  = dual_solution['mu']
            _eq_constr   = eq_constr(solution, args)
            _ineq_constr = ineq_constr(solution, args)
            return loss(solution, args) \
                + np.sum(lam * _eq_constr + c*0.5 * (_eq_constr)**2) \
                + (1/c)*0.5 * np.sum(np.maximum(0., mu + c*_ineq_constr)**2 - mu**2)

        val_dldx = jit(value_and_grad(lagrangian))
        _dldlam = jit(grad(lagrangian, argnums=1))
        gamma=0.9
        eps=1e-8
        @jit
        def step(solution, dual_solution, avg_sq_grad, args, c):
            _val, _dldx   = val_dldx(solution, dual_solution, args, c)
            for _key in solution:
                avg_sq_grad[_key] = avg_sq_grad[_key] * gamma + np.square(_dldx[_key]) * (1. - gamma)
                solution[_key] = solution[_key] - step_size * _dldx[_key] / np.sqrt(avg_sq_grad[_key] + eps)

            # _eps    = np.linalg.norm(_dldx['x'])
            # avg_sq_grad = avg_sq_grad * gamma + np.square(_dldx['x']) * (1. - gamma)
            # solution['x']   = solution['x'] - step_size * _dldx['x'] / np.sqrt(avg_sq_grad + eps)
            dual_solution['lam'] = dual_solution['lam'] + c*eq_constr(solution, args)
            dual_solution['mu']  = np.maximum(0, dual_solution['mu'] + c*ineq_constr(solution, args))
            return solution, dual_solution, avg_sq_grad, _val

        self.lagrangian      = lagrangian
        self.grad_lagrangian = val_dldx
        self.step = step

    def get_solution(self):
        return self.solution
        # return self._unravel(self._flat_solution)

    def solve(self, args=None, max_iter=100000, eps=1e-5):
        if args is None:
            args = self.def_args
        _eps = 1.0
        for k in range(max_iter):
            # self.solution, _val, self.avg_sq_grad = self.step(self.solution, args, self.avg_sq_grad, self.c)
            self.solution, self.dual_solution, self.avg_sq_grad, _val = self.step(self.solution, self.dual_solution, self.avg_sq_grad, args, self.c)
            self.c = 1.001*self.c
            if self._prev_val is None:
                self._prev_val = _val
            else:
                # print(_val)
                _eps = np.abs(_val - self._prev_val)
                self._prev_val = _val
            if _eps < eps:
                print('done in ', k, ' iterations')
                break

if __name__=='__main__':
    '''
        Example use case 
    '''
    def f(x, args=None) : return 13*x[0]**2 + 10*x[0]*x[1] + 7*x[1]**2 + x[0] + x[1]
    def g(x, args) : return np.array([2*x[0]-5*x[1]-2])
    def h(x, args) : return x[0] + x[1] -1

    x0 = np.array([.5,-0.3])
    opt = AugmentedLagrangian(x0,f,g,h, step_size=0.1)
    opt.solve(max_iter=1000)
    sol = opt.get_solution()
    print(f(sol['x']), sol['x'])