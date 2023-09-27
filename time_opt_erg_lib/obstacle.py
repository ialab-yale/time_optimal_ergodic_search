import jax.numpy as np
from jax import vmap
import matplotlib.pyplot as plt
import matplotlib.transforms as mpltf

def rot(th):
    return np.array(
        [[np.cos(th), -np.sin(th)],
        [np.sin(th), np.cos(th)]]
    )

class Obstacle(object): 
    def __init__(self, pos, half_dims, th, buff=0.1, p=4):
        self.pos = pos
        self.half_dims = half_dims
        self.dims = 2*self.half_dims
        self.buff = buff
        self.th = th
        self.rot = rot(self.th)
        self.rotT = self.rot.T
        self.inv_rot = lambda p: self.rotT@(self.pos - p)
        # self.min_dist = min_dist
        self.p = p
    
    def __getitem__(self, key):
        return self._obs_dict[key]

    def draw(self):
        rect = plt.Rectangle(self.pos-self.half_dims, 
                            self.dims[0], self.dims[1])
        return rect
    def distance3(self, x):
        return 1.0 - np.linalg.norm((x-self.pos)/(self.half_dims+self.buff), ord=2)

    def distance(self, x):
        # return 1.0 - np.linalg.norm(self.rot @ ((x-self.pos)/(self.half_dims+self.buff)), ord=4)
        return 1.0 - np.linalg.norm((self.rotT@(x-self.pos))/(self.half_dims+self.buff), ord=4)
        # dx = (x[0] - self.pos[0])/self.half_dims[0] 
        # dy = (x[1] - self.pos[1])/self.half_dims[1] 
        # return 1 - (dx**self.p + dy**self.p)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    X, Y = np.meshgrid(*[np.linspace(-2,2)]*2)
    pnts = np.vstack([X.ravel(), Y.ravel()]).T

    pos = np.array([0.,0.])
    dims = np.array([0.5/2, 0.7/2])
    min_dist = 1
    obs = Obstacle(pos, dims, min_dist)

    vals = vmap(obs.distance)(pnts)
    plt.contour(X, Y,vals.reshape(X.shape), levels=[-0.01,0.,0.01])
    plt.axis('equal')
    plt.show()