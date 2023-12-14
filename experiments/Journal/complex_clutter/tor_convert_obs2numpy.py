import numpy as onp

_tor_info1 = {
        'pos' : onp.array([1.75, 1.75, .5]), 
        'r1'  : .5,
        'r2'  : 0.28,
        'rot': 0.
    }

_tor_info2 = {
        'pos' : onp.array([.75, .75, .22]), 
        'r1'  : .5,
        'r2'  : 0.28,
        'rot': 0.
    }


onp.save('tor_obs_physical.npy', [_tor_info1, _tor_info2])

a = onp.load('tor_obs_physical.npy', allow_pickle=True)
print(a)