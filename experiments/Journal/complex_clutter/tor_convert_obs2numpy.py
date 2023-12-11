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
onp.save('tor_obs_physical.npy', [_tor_info1, _tor_info2])

a = onp.load('tor_obs_physical.npy', allow_pickle=True)
print(a)