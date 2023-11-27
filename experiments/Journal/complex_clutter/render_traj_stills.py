import numpy as np 
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap

from drone_env_viz.msg import Trajectory
from aircraft_viz import AgentViz
from env_viz import EnvViz
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from build_3D_solver import build_erg_time_opt_solver
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
    text_msg.pose.position.x = 0.
    text_msg.pose.position.y = 0.4
    text_msg.pose.position.z = 0.5

    traj_msg = Trajectory()
    traj_msg.name= agent_name + "_traj"


    solver, obs, args = build_erg_time_opt_solver()
    print("obs: ",end="")
    print(obs)

    env_viz = EnvViz(obs)
    agent_viz = AgentViz(agent_name)
    rate = rospy.Rate(10)

    print('Solving trajectory')
    solver.solve(args=args, max_iter=100, eps=1e-7)
    sol = solver.get_solution()
    print("Solved!!!")            

    text_msg.text = 'Optimal Time: {:.2f}'.format(sol['tf']) + '\n' + 'Maximum Ergodicity: {}'.format(args['erg_ub'])
    print('TAKE PICTURE NOW', text_msg.text)
    while not rospy.is_shutdown():
        solver.solve(args=args, max_iter=100, eps=1e-7)
        sol = solver.get_solution()
        with open('drone/test_trajs/test_traj.pkl', 'wb') as fp:
            pkl.dump(sol, fp)
        print(sol['tf'])
        # for i, _pt in enumerate(sol['x']):
        #     traj_msg.points[i].x = _pt[0]
        #     traj_msg.points[i].y = _pt[1]
        #     traj_msg.points[i].z = _pt[2]

        # traj_pub.publish(traj_msg)
        agent_viz.callback_trajectory(sol['x'])
        # text_pub.publish(text_msg)
        # br.sendTransform(
        #         (args['x0'][0], args['x0'][1], 0.35),
        #         (0.,0.,0.,1.),
        #         rospy.Time.now(),
        #         agent_name,
        #         "world"
        #     )
        env_viz.pub_env()
        rate.sleep()
        




## <---- below draws the objects with matplotlib ---->
# for obs in traj_opt.obs:
#     _patch = obs.draw()
#     plt.gca().add_patch(_patch)

# X, Y = np.meshgrid(*[np.linspace(wks[0],wks[1]) for wks in args['wrksp_bnds']])
# pnts = np.vstack([X.ravel(), Y.ravel()]).T

# _mixed_vals = np.inf * np.ones_like(X)
# for ob in obs:
#     _vals = np.array([ob.distance(pnt) for pnt in pnts]).reshape(X.shape)
#     _mixed_vals = np.minimum(_vals, _mixed_vals)

#     plt.contour(X, Y, _vals.reshape(X.shape), levels=[-0.01,0.,0.01])

# plt.plot(sol['x'][:,0], sol['x'][:,1],'g.')
# plt.plot(sol['x'][:,0], sol['x'][:,1])
# plt.show()