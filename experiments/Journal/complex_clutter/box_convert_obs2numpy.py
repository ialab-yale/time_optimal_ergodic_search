import numpy as onp

_box_info1 = {
        'pos' : onp.array([2.5, 0.5, 0.8]), 
        'half_dims' : onp.array([0.15, 0.15, 0.8]),
        'half_dims_barr' : onp.array([0.35, 0.35, 1.4]),
        'rot': 0.
    }

_box_info2 = {
        'pos' : onp.array([.5, 2.5, 0.8]), 
        'half_dims' : onp.array([0.15, 0.15, 0.8]),
        'half_dims_barr' : onp.array([0.35, 0.35, 1.4]),
        'rot': 0.
    }

_box_info3 = {
        'pos' : onp.array([1.75, 2.5, 1.]), 
        'half_dims' : onp.array([0.13, 0.13, 1.0]),
        'half_dims_barr' : onp.array([0.25, 0.25, 1.4]),
        'rot': 0.
    }

onp.save('box_obs_physical.npy', [_box_info1, _box_info2])

a = onp.load('box_obs_physical.npy', allow_pickle=True)
print(a)