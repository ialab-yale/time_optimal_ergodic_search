import numpy as onp

_box_info1 = {
        'pos' : onp.array([2.5, 0.5, 1.]), 
        'half_dims' : onp.array([0.2, 0.2, 1.0]),
        'half_dims_barr' : onp.array([0.35, 0.35, 1.4]),
        'rot': 0.
    }

_box_info2 = {
        'pos' : onp.array([.5, 2.5, 1.]), 
        'half_dims' : onp.array([0.2, 0.2, 1.0]),
        'half_dims_barr' : onp.array([0.35, 0.35, 1.4]),
        'rot': 0.
    }

# _tor_info2 = {
#         'pos' : onp.array([4.5, 6., 3.5]), 
#         'r1'  : 1.,
#         'r2'  : 0.25,
#         'rot': 0.
#     }

# _tor_info3 = {
#     'pos' : onp.array([2., 7.5, 1.]), 
#     'r1'  : 1.,
#     'r2'  : 0.25,
#     'rot': 0.
# }

# _tor_info4 = {
#     'pos' : onp.array([2., 2., 1.]), 
#     'r1'  : 1.,
#     'r2'  : 0.25,
#     'rot': 0.
# }

# _tor_info5 = {
#     'pos' : onp.array([6., 7.5, 1.]), 
#     'r1'  : 1.,
#     'r2'  : 0.25,
#     'rot': 0.
# }

# _tor_info6 = {
#     'pos' : onp.array([6., 2., 1.]), 
#     'r1'  : 1.,
#     'r2'  : 0.25,
#     'rot': 0.
# }

# _tor_info7 = {
#     'pos' : onp.array([2., 7.5, 5.]), 
#     'r1'  : 1.,
#     'r2'  : 0.25,
#     'rot': 0.
# }

# _tor_info8 = {
#     'pos' : onp.array([2., 2., 5.]), 
#     'r1'  : 1.,
#     'r2'  : 0.25,
#     'rot': 0.
# }

# _tor_info9 = {
#     'pos' : onp.array([6., 7.5, 5.]), 
#     'r1'  : 1.,
#     'r2'  : 0.25,
#     'rot': 0.
# }

# _tor_info10 = {
#     'pos' : onp.array([6., 2., 5.]), 
#     'r1'  : 1.,
#     'r2'  : 0.25,
#     'rot': 0.
# }

# onp.save('obs.npy', [_tor_info1,_tor_info2,_tor_info3,_tor_info4,_tor_info5,_tor_info6,_tor_info7,_tor_info8,_tor_info9,_tor_info10])
onp.save('box_obs_physical.npy', [_box_info1, _box_info2])

a = onp.load('box_obs_physical.npy', allow_pickle=True)
print(a)