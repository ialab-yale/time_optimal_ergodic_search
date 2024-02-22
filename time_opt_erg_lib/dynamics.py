import jax.numpy as np
import numpy as onp



class SingleIntegrator2D(object):
    def __init__(self) -> None:
        self.dt = 0.1
        self.n = 2
        self.m = 2
        B = np.array([
            [1.,0.],
            [0.,1.]
        ])
        def dfdt(x, u):
            return B@u
        def f(x, u):
            return x + self.dt*B@u
        self.f = f
        self.dfdt = dfdt

class SingleIntegrator3D(object):
    def __init__(self) -> None:
        self.dt = 0.1
        self.n = 3
        self.m = 3
        B = np.array([
            [1.,0.,0.],
            [0.,1.,0.],
            [0.,0.,1.]
        ])
        def dfdt(x, u):
            return B@u
        def f(x, u):
            # B = np.array([
            #     [np.cos(x[2]), 0.,],
            #     [np.sin(x[2]), 0.],
            #     [0., 1.]
            # ])
            return x + self.dt*B@u
        self.f = f
        self.dfdt = dfdt

class DoubleIntegrator(object):
    def __init__(self, dim=2) -> None:
        self.dt = 0.1
        self.n = dim
        self.m = dim
        A = np.eye(dim*2, dim*2, k=dim)
        # A = np.array([
        #     [0., 0., 1.0, 0.],
        #     [0., 0., 0.0, 1.],
        #     [0., 0., 0.0, 0.],
        #     [0., 0., 0.0, 0.]
        # ])
        B = np.eye(dim*2, dim, -dim)
        # B = np.array([
        #     [0., 0.],
        #     [0., 0.],
        #     [1., 0.],
        #     [0., 1.]
        # ])
        def dfdt(x, u):
            return A@x + B@u

        def f(x, u):
            # B = np.array([
            #     [np.cos(x[2]), 0.,],
            #     [np.sin(x[2]), 0.],
            #     [0., 1.]
            # ])
            return x + self.dt*B@u
        self.f = f
        self.dfdt = dfdt

class NDDoubleIntegrator(object):
    def __init__(self,ndim) -> None:
        self.dt = 0.1
        self.n = ndim
        self.m = ndim

        def dfdt(x, u):
            return np.concatenate((x[ndim:],u),axis=-1)
        self.dfdt = dfdt

# class KinematicBicycle(object):
#     def __init__(self) -> None:
#         self.dt = 0.1
#         self.n = 3
#         self.m = 2 
#         self.l = 1.0
#         def dfdt(x, u):
#             v = u[0]
#             w = u[1]
#             return np.array([
#                 v * np.cos(x[2]),
#                 v * np.sin(x[2]),
#                 w
#             ]) 
#         def f(x, u):
#             return x + self.dt * dfdt(x, u)
#         self.f      = f
#         self.dfdt   = dfdt

class KinematicUnicycle(object):
    def __init__(self) -> None:
        self.dt = 0.1
        self.n = 3
        self.m = 2
        self.l = 1.0
        def dfdt(x, u):
            v = u[0]
            w = u[1]
            return np.array([
                v * np.cos(x[2]),
                v * np.sin(x[2]),
                w
            ]) 
        def f(x, u):
            return x + self.dt * dfdt(x, u)
        self.f      = f
        self.dfdt   = dfdt

class SingleIntegratorBearing2D(object):
    def __init__(self) -> None:
        self.dt = 0.1
        self.n = 3
        self.m = 3
        B = np.array([
            [1.,0.,0.],
            [0.,1.,0.],
            [0.,0.,1.]
        ])
        def dfdt(x, u):
            return B@u
        def f(x, u):
            return x + self.dt*B@u
        self.f = f
        self.dfdt = dfdt


class ThreeDAirCraftModel(object):
    def __init__(self) -> None:
        self.dt = 0.1
        self.n = 5
        self.m = 3 
        def dfdt(x, u):
            v  = u[0]
            w1 = u[1]
            w2 = u[2] 
            return np.array([
                v * np.cos(x[4]) * np.cos(x[3]),
                v * np.sin(x[4]) * np.cos(x[3]),
                v * np.sin(x[3]),
                w1, 
                w2
            ]) 
        def f(x, u):
            return x + self.dt * dfdt(x, u)
        self.f      = f
        self.dfdt   = dfdt



# Drone constants 
_n = 18
_m = 4
# _dt = 1.0/200.0
# J = onp.diag([0.04, 0.0375, 0.0675])
J = onp.diag([0.025, 0.021, 0.043])

INV_J = onp.linalg.inv(J)

KT = 0.6
KM = 0.022
OMEGA_MAX = 10.
ARM_LENGTH = 0.15
MASS = 0.85
INV_MASS = 1.0/MASS
SQRT2 = onp.sqrt(2)

e_z = onp.array([0.,0.,1.])

g_vec = -9.81 * e_z

D = onp.diag([0.01,0.01,0.1])*0.

def hat(w):
    return np.array([
        [0.,      -w[2], w[1]],
	    [w[2],    0.,   -w[0]],
	    [-w[1],   w[0],   0.]]
    )

class DroneDynamics2(object):

    def __init__(self) -> None:
        self.n = _n 
        self.m = _m
        def dfdt(x, u):
            u = np.clip(u, 0., 7.)
            # g = x[0:16].reshape((4,4))
            # R, p = trans_to_Rp(g)
            R = x[:9].reshape((3,3))
            p = x[9:12]
            w = x[12:15]
            v = x[15:]
            # twist   = x[16:]

            f_t = u[0] + u[1] + u[2] + u[3]
            tau = np.array([
                            ARM_LENGTH * (u[0] + u[1] - u[2] - u[3])/SQRT2,
                            ARM_LENGTH * (-u[0] + u[1] + u[2] - u[3])/SQRT2,
                            KM * (u[0] - u[1] + u[2] - u[3])
            ])

            wdot = INV_J @ (tau - np.cross(w, J@w))
            vdot = g_vec + INV_MASS * R @ (e_z*f_t) #- R@D@R.T@v 
            pdot = v 
            dR = (R @ hat(w)).ravel()
            # dg = (g @ hat(twist)).ravel()
            return np.concatenate([dR, pdot, wdot, vdot])
        self.dfdt = dfdt
        # self.fdx = jit(jacfwd(self.f, argnums=0))
        # self.fdu = jit(jacfwd(self.f, argnums=1))
# if __name__=='__main__':
#     drone = DroneDynamics()

#     R = np.eye(3)
#     p = np.ones(3)*2
#     w = np.ones(3)*3
#     v = np.ones(3)*4
#     x = np.concatenate([R.ravel(), p, w, v])
#     print(x)
#     R = x[:9].reshape((3,3))
#     p = x[9:12]
#     omega   = x[12:15]
#     v       = x[15:]
#     print(R, p, omega, v)
#     u = np.zeros(4)

#     drone.f(x, u)


# Drone constants 
_n = 18
_m = 4
# _dt = 1.0/200.0
# J = onp.diag([0.04, 0.0375, 0.0675])
J = onp.diag([0.025, 0.021, 0.043])

INV_J = onp.linalg.inv(J)

KT = 0.6
KM = 0.022
OMEGA_MAX = 10.
ARM_LENGTH = 0.15
MASS = 0.85
INV_MASS = 1.0/MASS
SQRT2 = onp.sqrt(2)

e_z = onp.array([0.,0.,1.])

g_vec = -9.81 * e_z

D = onp.diag([0.01,0.01,0.01])*1.0

def hat(w):
    return np.array([
        [0.,      -w[2], w[1]],
	    [w[2],    0.,   -w[0]],
	    [-w[1],   w[0],   0.]]
    )

class DroneDynamics(object):

    def __init__(self) -> None:
        self.n = _n 
        self.m = _m
        def dfdt(x, u):
            # u = np.clip(u, 0., 7.)
            # g = x[0:16].reshape((4,4))
            # R, p = trans_to_Rp(g)
            p = x[:3]
            R = x[3:12].reshape((3,3))
            w = x[12:15]
            v = x[15:]
            # twist   = x[16:]

            f_t = u[0] + u[1] + u[2] + u[3]
            tau = np.array([
                            ARM_LENGTH * (u[0] + u[1] - u[2] - u[3])/SQRT2,
                            ARM_LENGTH * (-u[0] + u[1] + u[2] - u[3])/SQRT2,
                            KM * (u[0] - u[1] + u[2] - u[3])
            ])

            wdot = INV_J @ (tau - np.cross(w, J@w))
            vdot = g_vec + INV_MASS * R @ (e_z*f_t) #- R@D@R.T@v 
            pdot = v 
            dR = (R @ hat(w)).ravel()
            # dg = (g @ hat(twist)).ravel()
            return np.concatenate([pdot, dR, wdot, vdot])
        self.dfdt = dfdt