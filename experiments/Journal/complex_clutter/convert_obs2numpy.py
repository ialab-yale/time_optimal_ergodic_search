import yaml
import io
import numpy as np

_tor_info = {
        'pos' : np.array([1., 1., 1.]), 
        'r1'  : 4.,
        'r2'  : 1.,
        'rot': 0.
    }

np.save('obs.npy', [_tor_info])

a = np.load('obs.npy', allow_pickle=True)
print(a[0])