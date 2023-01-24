import numpy as np
import pickle as pkl

scale = 100.0
obstacle_x  = np.array([2.84, 1.26, 2.62, 2.50, 3.05,  1.77, 0.64, 0.82, 0.73,  1.03, 1.63,  1.67, 1.87])          # obstacle x position of center
obstacle_y  = np.array([1.82, 3.31, 0.08, 2.85, 1.08, -0.47, 2.80, 1.95, 1.04, -0.07, 2.52,  0.41, 1.64])          # obstacle y position of center
rot         = -np.array([0.00, 0.00, 0.00, 0.00, 45.0,  0.00, 45.0, 0.00, 45.0,  0.00, 0.00, 315.0, 0.00])*np.pi/180.0          # obstacle angle in degrees
obstacle_l  = np.array([19.8, 30.0, 20.0, 20.0, 13.5,  13.5, 20.0, 13.5, 13.5,  13.5, 13.5,  49.6, 47.0])/scale    # obstacle length in meters
obstacle_w  = np.array([10.1, 30.0, 20.0, 20.0, 13.5,  13.5, 20.0, 13.5, 13.5,  13.5, 13.5,  10.1, 47.0])/scale    # obstacle width in meters

obs_info = {
    # 'obs 1': {'pos' : [2.84, 1.82], 'width/2': 0.01*19.8/2.0}
    # 'obs_'
}
for i,(x,y,l,w,r) in enumerate(zip(obstacle_x, obstacle_y, obstacle_l, obstacle_w, rot)):
    obs_name = 'obs_{}'.format(i)
    obs_info.update(
        {obs_name : {'pos' : [x,y], 'half_dims' : [w/2.0, l/2.0], 'rot':r }}
    )

print(obs_info)

pkl.dump(obs_info, open("obs_info.pkl", "wb"))

for name in obs_info:
    print(name)