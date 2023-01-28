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
def sdf2cbf(dfdt, constr):
    return lambda x, u, alpha, dt : constr((x + dt * dfdt(x,u))[:2]) - (1.-alpha) * constr(x[:2])

# # Regular Inequality Constraints
# def sdf2cbf(f, constr):
#     return lambda x, u, alpha: constr(x)