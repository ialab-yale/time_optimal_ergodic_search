from typing import Callable,Any

from jax import jit,value_and_grad,Array,tree_map,vmap
from jax.tree_util import Partial
import jax
import jax.numpy as jnp
from jaxlie import manifold,SE3

def make_manifold_vjp(fun:Callable,has_aux=False,reduce_axes=()):
    def tangent_fun(primal_args,*tangent_args):
        return fun(*manifold.rplus(primal_args,tangent_args))
    @jit
    def manifold_vjp(*primals):
        tangent_args=manifold.zero_tangents(primals)
        return jax.vjp(Partial(tangent_fun,primals),*tangent_args,has_aux=has_aux,reduce_axes=reduce_axes)
    return manifold_vjp
@jit
def vector_matrix_mul_batched(vector:Array,matrix:Array):
    '''
    left-multiply a batch of vectors into a batch of matrices, v.T@M

    Parameters: vector : (...,m) Array
                    leading dimensions are batch dimension(s)
                matrix : (...,m,n) Array
                    leading dimensions are the same batch dimension(s)
    Returns:    out_vector : (...,m) Array
                    leading dimensions are the same batch dimension(s)
    '''
    pass
@manifold._deltas._naive_auto_vmap
def pullback(cotangent:SE3,primal:SE3):
    Jparam_tangent=manifold.rplus_jacobian_parameters_wrt_delta(primal)
    return Jparam_tangent.T@cotangent.wxyz_xyz
@jit
def pytree_manifold_pullback(cotangents,primals):
    return manifold._tree_utils._map_group_trees(pullback,lambda vjped,primal:vjped,cotangents,primals)

def transform_vjp_with_pullback(raw_vjp_fun,primals):
    def manifold_vjp_fun(cotangent,primals):
        raw_vjped=raw_vjp_fun(cotangent)
        return pytree_manifold_pullback(raw_vjped,primals)
    return Partial(manifold_vjp_fun,primals=primals)
def manifold_vjp(fun:Callable,*primals,has_aux=False,reduce_axes=()):
    '''
    same as jax.vjp, but handles Lie groups via jaxlie
    '''
    value,raw_vjp_fun=jax.vjp(fun,*primals,has_aux=has_aux,reduce_axes=reduce_axes)
    manifold_vjp_fun=transform_vjp_with_pullback(raw_vjp_fun,primals)
    return value,manifold_vjp_fun
def non_manifold_vjp(fun:Callable,*primals,has_aux=False,reduce_axes=()):
    value,raw_vjp_fun=jax.vjp(fun,*primals,has_aux=has_aux,reduce_axes=reduce_axes)
    return value,raw_vjp_fun
@jit
def add_up_grad(dLdx:dict,dLdg_dgdx:dict,dLdh_dhdx:dict):
    for key in dLdx:
        dLdx[key]+=dLdg_dgdx[key]+dLdh_dhdx[key]
    return dLdx

@jit
def nan_test(*args):
    return manifold._tree_utils._map_group_trees(lambda x:jnp.any(jnp.isnan(x.parameters())),lambda x:jnp.any(jnp.isnan(x)),args)
class Optimizer:
    '''
    Augmented Lagrangian trajectory opt base class for jax jittable loss, equality, and inequality constraints

    Solves problems of the form
    min J(x,s) s.t.
     x
        g(x,s)=0
        h(x,s)<=0
    '''
    def __init__(self,loss_function:Callable[[dict,Any],float],equality_constraints:Callable[[dict,Any],Array],ineq_constraints:Callable[[dict,Any],Array],step_size=1e-3,c=1.0,verbose=True):
        '''
        Parameters: loss_function : Callable[[dict,Any],float]
                        given solution dictionary x and fixed arguments s, return cost J(x,s). Must be jax jittable
                    equality_constraints : Callable[[dict,Any],Array]
                        given solution dictionary x and fixed arguments s, return equality constraints g(x,s). Must be jax jittable.
                    ineq_constraints : Callable[[dict,Any],Array]
                        given solution dictionary x and fixed arguments s, return inequality constraints h(x,s). Must be jax jittable. Recall that h(x,s)<=0 is the feasible region.
                    step_size : float, default 1e-3
                        step size for optimizer
                    c : float, default 1.0
                        initial value of penalty coefficient in the augmented Lagrangian
                    verbose : bool, default True
                        if True, print every hundredth iteration
        '''
        self.pullback=lambda cotangents,primals:cotangents
        self.verbose=verbose
        self.loss=loss_function
        self.eq_constr=equality_constraints
        self.ineq_constr=ineq_constraints
        self.step_size=step_size
        self.c=jnp.array(c)
        self.lagrangian=self.make_lagrangian()
        self.lagrangian_val_and_partials=value_and_grad(self.lagrangian,argnums=(0,1,2))
        self.lagrangian_value_and_grad=self.make_lagrangian_value_and_grad()
        self.solver_step=jit(self.make_jittable_solver_step())
        self.increment_solution=lambda sol,inc:sol+inc
        self.zero_tangents=lambda sol:{key:jnp.zeros_like(sol[key]) for key in sol}
        self.vjp=jax.vjp
        
    def make_stepper(self):
        '''
        after instatiating an Optimizer subclass, assign the output of this function to .step before calling .solve()
        '''
        pass
    def make_lagrangian_value_and_grad(self):
        def lagrangian_value_and_grad(solution:dict,eq_violation:Array,ineq_violation:Array,lam:Array,mu:Array,c:float,eq_vjp:Callable,ineq_vjp:Callable,args:Any):
            '''
            computes augmented lagrange minimization objective:
            min J(x,s)+dot(lambda,g(x,s))+c*dot(g(x,s),g(x,s))/2+sum(max(0,mu_i+c*h_i(x,s))**2-mu_i**2)/c/2
                x
            and its gradient with respect to solution, INCLUDING the contribution from the dependence of eq_violation and ineq_violation on solution
            Parameters: solution : dict with string keys and jax compatible values
                            specifies the optimization variables
                        eq_violation : (n_equality,) jnp float Array
                                value of the equality constraint functions (which is 0 when constraint is satisfied)
                        ineq_violation : (n_inequality,) jnp float Array
                            value of the inequality constraint functions (which is negative when the constraint is satisfied)
                        lam : (n_equality,) jnp float array
                            multipliers for the n_equality equality constraints
                        mu : (n_inequality,) jnp float array
                            multipliers for the n_inequality inequality constraints
                        c : float
                            penalty coefficient
                        eq_vjp : Callable
                            vector-Jacobian product function for equality constraints, returned by jax.vjp(self.eq_constr,solution)
                        ineq_vjp : Callable
                            vector-Jacobian product function for inequality constraints, returned by jax.vjp(self.ineq_constr,solution)
                        arguments : Any
                            fixed arguments to J,g,h, if any
            Return:     L : (,) jnp float Array
                            augmented lagrangian objective value
                        dLdx : dict with string keys and jax compatible values (i.e. in the tangent space of solution)
                            gradient of L wrt solution
            '''
            L,grads=self.lagrangian_val_and_partials(solution,eq_violation,ineq_violation,lam,mu,c,args)
            dLdx,dLdg,dLdh=grads
            dLdg_dgdx=self.pullback(eq_vjp(dLdg),(solution,args))[0]
            dLdh_dhdx=self.pullback(ineq_vjp(dLdh),(solution,args))[0]
            return L,add_up_grad(dLdx,dLdg_dgdx,dLdh_dhdx)
        return lagrangian_value_and_grad

    def make_lagrangian(self):
        def lagrangian(solution:dict,eq_violation:Array,ineq_violation:Array,lam:Array,mu:Array,c:float,args:Any):
            '''
            computes augmented lagrange minimization objective: 
            min J(x,s)+dot(lambda,g(x,s))+c*dot(g(x,s),g(x,s))/2+sum(max(0,mu_i+c*h_i(x,s))**2-mu_i**2)/c/2
              x
            Parameters: solution : dict with string keys and jax compatible values
                            specifies the optimization variables
                        eq_violation : (n_equality,) jnp float Array
                            value of the equality constraint functions (which is 0 when constraint is satisfied)
                        ineq_violation : (n_inequality,) jnp float Array
                            value of the inequality constraint functions (which is negative when the constraint is satisfied)
                        lam : (n_equality,) jnp float array
                            multipliers for the n_equality equality constraints
                        mu : (n_inequality,) jnp float array
                            multipliers for the n_inequality inequality constraints
                        c : float
                            penalty coefficient
                        arguments : Any
                            fixed arguments to J,g,h, if any
            Return:     L : (,) jnp float Array
                            augmented lagrangian objective value
            '''
            return self.loss(solution,args)+jnp.sum(lam*eq_violation+c*.5*eq_violation**2)+1/c*.5*jnp.sum(jnp.maximum(0.,mu+c*ineq_violation)**2-mu**2)
        return lagrangian
    def make_jittable_solver_step(self):
        def jittable_solver_step(solution,avg_sq_grad,eq_violation,ineq_violation,lam,mu,c,eq_vjp,ineq_vjp,args):
            value,lagrangian_grad=self.lagrangian_value_and_grad(solution,eq_violation,ineq_violation,lam,mu,c,eq_vjp,ineq_vjp,args)
            new_solution,new_avg_sq_grad,new_eq_violation,new_ineq_violation,new_eq_vjp,new_ineq_vjp=self.step(solution,lagrangian_grad,avg_sq_grad,c,args)
            new_lam=lam+c*new_eq_violation
            new_mu=jnp.maximum(0,mu+c*new_ineq_violation)
            return value,new_solution,new_avg_sq_grad,new_eq_violation,new_ineq_violation,new_lam,new_mu,new_eq_vjp,new_ineq_vjp
        return jittable_solver_step
    def print_iter(self,iteration:int,loss:float,lagrangian_value:float,abs_lagrangian_change:float,constraint_satisfaction:float):
        print(f"Iter {iteration}: Loss: {loss} Lagrangian: {lagrangian_value} Absolute Change: {abs_lagrangian_change} Constraint: {constraint_satisfaction}")
    def nan_test(self,iteration:int):
        '''
        test if any entries in the primal or dual variables have become NaN, logging if True
        '''
        sol_nans,lam_nan,mu_nan=nan_test(self.solution,self.lam,self.mu)
        stop=False
        for key in self.solution:
            if sol_nans[key]:
                print(f"Iter {iteration}: {key} contains NaN")
                stop=True
        if lam_nan:
            print(f"Iter {iteration}: lam contains NaN")
            stop=True
        if mu_nan:
            print(f"Iter {iteration}: mu contains NaN")
            stop=True
        return stop
    def solve(self,initial_guess:dict,max_iter=10000,lagrangian_convergence_tol=1e-5,feasibility_tol=5e-3,alpha=1.001,args=None):
        '''
        compute solution and store in self.solution,self.lam,self.mu,self.avg_sq_grad

        Parameters: initial_giess : dict with string keys and jax compatible values
                        specify initial optimization variable values
                    max_iter : int, default 10000
                        maximum number of optimizer steps
                    lagrangian_convergence_tol : float, default 1e-5
                        optimization only stops if absolute Lagrangian change in last iter was smaller than this
                    feasibility_tol : float, default 5e-3
                        optimization only stops if largest absolute constraint violation is smaller than this
                    alpha : float, default 1.001
                        increase penalty coefficient in lagrangian by this factor at each step
                    args : Any, default None
                        fixed arguments to J,g,h, if any
        '''
        self.solution=initial_guess

        c=self.c
        eps=2*lagrangian_convergence_tol
        previous_val=None

        eq_violation,eq_vjp=self.vjp(self.eq_constr,initial_guess,args)
        self.lam=jnp.zeros_like(eq_violation)
        ineq_violation,ineq_vjp=self.vjp(self.ineq_constr,initial_guess,args)
        self.mu=jnp.zeros_like(ineq_violation)

        self.avg_sq_grad=self.zero_tangents(self.solution)

        for k in range(max_iter):
            value,self.solution,self.avg_sq_grad,eq_violation,ineq_violation,self.lam,self.mu,eq_vjp,ineq_vjp=self.solver_step(self.solution,self.avg_sq_grad,
                                                                                                                                      eq_violation,ineq_violation,
                                                                                                                                      self.lam,self.mu,c,
                                                                                                                                      eq_vjp,ineq_vjp,args)
            stop=self.nan_test(k)
            constraint_satisfaction=max(jnp.max(jnp.abs(eq_violation)),jnp.max(ineq_violation))
            if stop:
                self.print_iter(k,self.loss(self.solution,args),value,eps,constraint_satisfaction)
                break
            c=alpha*c
            if previous_val is None:
                previous_val=value
            else:
                eps=jnp.abs(value-previous_val)
                previous_val=value
            if eps<lagrangian_convergence_tol and constraint_satisfaction<feasibility_tol:
                if self.verbose:
                    self.print_iter(k,self.loss(self.solution,args),value,eps,constraint_satisfaction)
                    print('done in ',k,' iterations')
                return
            if self.verbose:
                if max_iter<=100 or k%int(max_iter/100)==0:
                    self.print_iter(k,self.loss(self.solution,args),value,eps,constraint_satisfaction)
        print('unsuccessful, tol: ', eps)
        return

class RMSProp(Optimizer):
    '''
    Augmented Lagrangian trajectory opt using RMSProp for jax jittable loss, equality, and inequality constraints

    Solves problems of the form
    min J(x,s) s.t.
     x
        g(x,s)=0
        h(x,s)<=0

    original credit Ian Abraham et al. for RMSProp implementation
    '''
    def make_stepper(self,gamma=.9,eps=1e-8):
        '''
        return jax-jitted optimization step

        Parameters: gamma : float, default 0.9
                        fraction of avg_sq_grad at previous step to retain at current step
                    eps : float, default 1e-8
                        small constant to add to avg_sq_grad in denominator of increment
        Returns:    jitted step function
        '''
        @jit
        def rmsprop_step(solution:dict,lagrangian_grad:dict,avg_sq_grad:dict,c:float,args:Any):
            '''
            return updated solution, dual solution (lam and mu), avg_sq_grad, and constraint violations

            Parameters: solution : dict with string keys and jax compatible values
                            specifies the optimization variables
                        lagrangian_grad : dict with string keys and jax compatible values (i.e. in the tangent space of solution)
                            gradient of augmented Lagrangian wrt solution
                        avg_sq_grad : dict with string keys and jax compatible values in tangent space of solution's values
                            records the moving average squared gradient magnitudes
                        c : float
                            penalty coefficient
                        args : Any, default None
                            fixed arguments to J,g,h, if any
            Returns:    solution : dict with string keys and jax compatible values
                            updated optimization variables
                        avg_sq_grad : dict with string keys and jax compatible values in tangent space of solution's values
                            updated moving average squared gradient magnitudes
                        eq_violation : (n_equality,) jnp float Array
                            value of the equality constraint functions (which is 0 when constraint is satisfied)
                        ineq_violation : (n_inequality,) jnp float Array
                            value of the inequality constraint functions (which is negative when the constraint is satisfied)
            '''
            for key in solution:
                avg_sq_grad[key]=avg_sq_grad[key]*gamma+jnp.square(lagrangian_grad[key])*(1-gamma)
                solution[key]=self.increment_solution(solution[key],-self.step_size*lagrangian_grad[key]/jnp.sqrt(avg_sq_grad[key]+eps))
            eq_violation,eq_vjp=self.vjp(self.eq_constr,solution,args)
            ineq_violation,ineq_vjp=self.vjp(self.ineq_constr,solution,args)
            return solution,avg_sq_grad,eq_violation,ineq_violation,eq_vjp,ineq_vjp
        return rmsprop_step
    
class SE3Optimization(Optimizer):
    '''
    Mix this class in to an Optimizer to handle state spaces that include SE(3) using jaxlie
    '''
    def __init__(self, loss_function: Callable[[dict, Any], float], equality_constraints: Callable[[dict, Any], Array], ineq_constraints: Callable[[dict, Any], Array], step_size=0.001, c=1.0,verbose=True):
        '''
        Parameters: loss_function : Callable[[dict,Any],float]
                        given solution dictionary x and fixed arguments s, return cost J(x,s). Must be jax jittable
                    equality_constraints : Callable[[dict,Any],Array]
                        given solution dictionary x and fixed arguments s, return equality constraints g(x,s). Must be jax jittable.
                    ineq_constraints : Callable[[dict,Any],Array]
                        given solution dictionary x and fixed arguments s, return inequality constraints h(x,s). Must be jax jittable. Recall that h(x,s)<=0 is the feasible region.
                    step_size : float, default 1e-3
                        step size for optimizer
                    c : float, default 1.0
                        initial value of penalty coefficient in the Lagrangian
                    verbose : bool, default True
                        if True, print every hundredth iteration
        '''
        self.pullback=pytree_manifold_pullback
        self.verbose=verbose
        self.loss=loss_function
        self.eq_constr=equality_constraints
        self.ineq_constr=ineq_constraints
        self.step_size=step_size
        self.c=jnp.array(c)
        self.lagrangian=self.make_lagrangian()
        self.lagrangian_val_and_partials=manifold.value_and_grad(self.lagrangian,argnums=(0,1,2))
        self.lagrangian_value_and_grad=self.make_lagrangian_value_and_grad()
        self.solver_step=jit(self.make_jittable_solver_step())
        self.increment_solution=manifold.rplus
        self.zero_tangents=manifold.zero_tangents
        self.vjp=jax.vjp

    # def nan_test(self, iteration):
    #     '''
    #     test if any entries in the primal or dual variables have become NaN, logging if True
    #     '''
    #     stop=False
    #     for key in self.solution:
    #         if isinstance(self.solution[key],SE3) and jnp.any(jnp.isnan(self.solution[key].wxyz_xyz)):
    #             print(f"Iter {iteration}: {key} contains NaN")
    #             stop=True
    #     other_stop= super().nan_test(iteration)
    #     return stop or other_stop

class RMSPropOnSE3(RMSProp,SE3Optimization):
    '''
    Augmented Lagrangian trajectory opt using RMSProp for jax jittable loss, equality, and inequality constraints with states partially in SE(3)

    Solves problems of the form
    min  J(x,s) s.t.
    x\inX
         g(x,s)=0
         h(x,s)<=0
    for X=(SE(3))^m X R^n
    original credit Ian Abraham et al. for RMSProp implementation
    '''
    pass

def main():
    '''
    Example use case 
    '''
    def f(x, args=None) : return 13*x['x'][0]**2 + 10*x['x'][0]*x['x'][1] + 7*x['x'][1]**2 + x['x'][0] + x['x'][1]
    def g(x, args) : return jnp.array([2*x['x'][0]-5*x['x'][1]-2])
    def h(x, args) : return x['x'][0] + x['x'][1] -1

    x0 = {'x':jnp.array([.5,-0.3])}
    opt = RMSProp(f,g,h, step_size=0.1)
    opt.step=opt.make_stepper()
    opt.solve(initial_guess=x0,max_iter=1000)
    sol = opt.solution
    print(f(sol), sol['x'])

if __name__=='__main__':
    main()