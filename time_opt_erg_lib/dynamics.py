import jax.numpy as np



class SingleIntegrator(object):
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

class KinematicBicycle(object):
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