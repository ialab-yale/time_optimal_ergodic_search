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
    def __init__(self) -> None:
        self.dt = 0.1
        self.n = 2
        self.m = 2
        A = np.array([
            [0., 0., 1.0, 0.],
            [0., 0., 0.0, 1.],
            [0., 0., 0.0, 0.],
            [0., 0., 0.0, 0.]
        ])
        B = np.array([
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])
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