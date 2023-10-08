import numpy as np
import pickle as pkl

scale = 100.0
obstacle_x  = np.array([0.5, 0.5])          # obstacle x position of center
obstacle_y  = np.array([0.5, 0.5])          # obstacle y position of center
rot         = -np.array([0.00, 0.00])*np.pi/180.0          # obstacle angle in degrees
obstacle_l  = np.array([50., 25.0])/scale    # obstacle length in meters
obstacle_w  = np.array([20., 10.0])/scale    # obstacle width in meters

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