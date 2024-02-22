import numpy as np 
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
import math

from drone_env_viz.msg import Trajectory
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from distributions import ExpectedInformation, TargetBelief 
from sensor_model import GaussianSensorModel, LocalizationSensorModel, get_sensor_ck
from build_solver import build_erg_time_opt_solver
import pickle as pkl

import rospy
import tf


if __name__ =="__main__":

    rospy.init_node('topt_solver_node')

    agent_name="dude"

    br = tf.TransformBroadcaster()
    traj_pub = rospy.Publisher(agent_name + '/planned_traj', Trajectory, queue_size=10)

    text_pub = rospy.Publisher(agent_name + '/viz/text', Marker, queue_size=10)
    text_msg = Marker()
    text_msg.header.frame_id = "world"
    text_msg.color.a = 1.0
    text_msg.color.r = 0.0
    text_msg.color.g = 0.0
    text_msg.color.b = 0.0
    text_msg.scale.z = 0.1
    text_msg.type = Marker.TEXT_VIEW_FACING
    text_msg.text = "Testing"
    text_msg.pose.position.x = 0.5
    text_msg.pose.position.y = 1.2
    text_msg.pose.position.z = 0.8

    traj_msg = Trajectory()
    traj_msg.name= agent_name + "_traj"

    args = {
        'N' : 81, 
        'x0' : np.array([0.944, 0.055]),
        'xf' : np.array([0.055, 0.944]),
        'erg_ub' : 0.2,
        'alpha' : 0.5,
        'wrksp_bnds' : np.array([[0.,1.],[0.,1.]])
    }

    expect_info    = ExpectedInformation(args['wrksp_bnds'])  # May need this to be a 3D pdf with bearing as the third dimension
    target_belief    = TargetBelief(args['wrksp_bnds'])
    target_space = target_belief._s
    sensor = LocalizationSensorModel(np.array([0.1, 0.1]), np.array([0,1]))
    prior = target_belief
    post = prior
    targets = np.array([[0.5, 0.5]])

    solver = build_erg_time_opt_solver(args, expect_info, sensor)
    sol = solver.get_solution()
    
    rate = rospy.Rate(10)
    traj_msg.points = [Point(_pt[0], _pt[1], 0.35) for _pt in sol['x']]

    print('publishing trajectory')

    erg_ub = 0.001270128
    eta = 1.
    sig = 1.
    args.update({'erg_ub': erg_ub})

    while True:
        solver.reset()
        solver.solve(args=args, max_iter=100000, eps=1e-6, alpha=1.002)
        sol = solver.get_solution()
        # with open('test.pkl', 'wb') as fp:
        #     pkl.dump(sol, fp)

        text_msg.text = 'Optimal Time: {:.2f}'.format(sol['tf']) + '\n' + 'Maximum Ergodicity: {}'.format(erg_ub)
        print(text_msg.text)

        # # Update the prior of target belief
        # Vk = sensor.compute_observation_array(sol, targets)
        # Upsilon = sensor.compute_truth(sol, targets)
        # p = (1/(math.sqrt(2*math.pi)*sig)) * np.exp((Vk-Upsilon)**2/(-2*sig**2))
        # post = eta * p@prior

        # Update the prior of target belief
        for i, _target in enumerate(target_space):
            p = 1.
            for x in sol:
                Upsilon = sensor.compute_truth(x, _target)
                Vk = sensor.compute_observation_array(x, targets)
                p *= (1/(math.sqrt(2*math.pi)*sig)) * np.exp((Vk-Upsilon)**2/(-2*sig**2))
                post.evals[i] = prior.evals[i] * p
        post.evals /= np.sum(post.evals)    # Check reassignments

        # Compute Fisher information

        # Update expected information map

        for _ in range(100):
            for i, _pt in enumerate(sol['x']):
                traj_msg.points[i].x = _pt[0]
                traj_msg.points[i].y = _pt[1]

            traj_pub.publish(traj_msg)
            text_pub.publish(text_msg)
            br.sendTransform(
                    (args['x0'][0], args['x0'][1], 0.35),
                    (0.,0.,0.,1.),
                    rospy.Time.now(),
                    agent_name,
                    "world"
                )
            expect_info.pub_map()
            rate.sleep()
        