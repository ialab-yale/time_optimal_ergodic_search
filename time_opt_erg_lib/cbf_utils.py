import jax.numpy as np

def dist_func(x):
    d1 = 0.1-np.linalg.norm(x[0,:] - x[1,:])    # pair-wise interactions amongst the drones 
    # d2 = 0.1-np.linalg.norm(x[1,:] - x[3,:])
    # d3 = 0.1-np.linalg.norm(x[3,:] - x[2,:])
    # d4 = 0.1-np.linalg.norm(x[0,:] - x[2,:])
    # d5 = 0.1-np.linalg.norm(x[0,:] - x[3,:])
    # d6 = 0.1-np.linalg.norm(x[1,:] - x[2,:])
    # dist = [d1.flatten(), d2.flatten(), d3.flatten(), d4.flatten(), d5.flatten(), d6.flatten()]
    dist = [d1.flatten()]
    return np.array(dist)

# CBF Inequality Constraints
# def sdf2cbf(dfdt, constr):
#     return lambda x, u, alpha, dt : constr((x + dt * dfdt(x,u))[:2]) - (1.-alpha) * constr(x[:2])
def sdf2cbf(f, constr):
    return lambda x, u, alpha: constr(f(x,u)) - (1.-alpha) * constr(x)

def sdf2cbfhole(f, constrout, constrin):
    return lambda x, u, alpha: np.maximum(constrout(f(x,u)) - (1.-alpha) * constrout(x), -(constrin(f(x,u)) - (1.-alpha) * constrin(x)))

def sdf3cbf(dfdt, constr):
    return lambda x, u, alpha, dt : constr((x + dt * dfdt(x,u))[:3]) - (1.-alpha) * constr(x[:3])

def sdf3cbfhole(dfdt, constrout, constrin):
    return lambda x, u, alpha, dt : np.maximum(constrout((x + dt * dfdt(x,u))[:3]) - (1.-alpha) * constrout(x[:3]), -(constrin((x + dt * dfdt(x,u))[:3]) - (1.-alpha) * constrin(x[:3])))

# # Regular Inequality Constraints
# def sdf2cbf(f, constr):
#     return lambda x, u, alpha: constr(x)