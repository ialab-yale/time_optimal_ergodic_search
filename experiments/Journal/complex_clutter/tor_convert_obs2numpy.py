import numpy as onp

_tor_info1 = {
        'pos' : onp.array([0.75, 1.75, .5]), 
        'r1'  : .5,
        'r2'  : 0.28,
        'rot': 0.
    }

_tor_info2 = {
        'pos' : onp.array([2.25, 1.75, .5]), 
        'r1'  : .5,
        'r2'  : 0.28,
        'rot': 0.
    }

_tor_info3 = {
        'pos' : onp.array([1.5, 1.75, 2.]), 
        'r1'  : .5,
        'r2'  : 0.28,
        'rot': 0.
    }

_tor_info4 = {
        'pos' : onp.array([2.25, 1.75, 2.]), 
        'r1'  : .5,
        'r2'  : 0.28,
        'rot': 0.
    }


onp.save('tor_obs.npy', [_tor_info1, _tor_info2, _tor_info3])

a = onp.load('tor_obs.npy', allow_pickle=True)
print(a)