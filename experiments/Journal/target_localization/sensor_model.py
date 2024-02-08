import numpy as np
import jax
from jax import vmap

class SensorModel():
    def __init__(self) -> None:
        self.compute_observation_array=vmap(vmap(self.observation_function,(None,0)),(0,None))
    def observation_function(self,state,query):
        pass

class GaussianSensorModel(SensorModel):
    '''
    Sensor model for k-dimensional sensor domain where observation intensity decays as a multi-variate uncorrelated gaussian
    '''
    def __init__(self,std,position_indices) -> None:
        '''
        Parameters: std : float or (k,) jax float array
                        standard deviation along each dimension
                    position_indices : (k,) jax int array
                        which indices in the robot state correspond to positions wrt the domain of the sensor (i.e. where is x,y,z located in the state vector?)
        '''
        self.std=std
        self.k=len(position_indices)
        self.variance=self.std**2
        self.position_indices=position_indices
        super().__init__()
    def observation_function(self, state, query):
        '''
        pdf intensity for the sensor at a query point if the sensor is located at a particular robot state

        Parameters: state : (n,) jax float array
                        robot state, m>=k
                    query : (k,) jax float array
                        position in the sensor domain
        Returns:    sensing likelihood : jax float
                        sensor probability mass at query if robot is at state (should be between 0 and 1, need not integrate to 1 across possible states and/or queries)
        '''
        pos=state[self.position_indices]
        err=query-pos-0.1
        return np.exp(np.dot(err,err/self.variance)/(-2))
    
def get_sensor_ck(trajectory, sensor_model, distribution, basis, tf, dt, twod=False):
    if twod:
        obs_array=sensor_model.compute_observation_array(trajectory,distribution._s2)
    else:
        obs_array=sensor_model.compute_observation_array(trajectory,distribution._s)    #matrix (time by mesh vertices) that represents the amount each vertex on the mesh is observed at each point in time along the trajectory

    obs_func = np.sum(obs_array, axis=0) * dt
    f = obs_func/np.sum(obs_func@basis.sphara_basis.massmatrix())
    # f = obs_func/np.sum(obs_func@basis.sphara_basis.triangsamples.massmatrix(mode='lumped'))'

    ck = basis.coefficients(f)
    ck = ck / basis.hk_list
    return ck