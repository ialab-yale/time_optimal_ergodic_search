#!/usr/bin/env python3

import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from drone_env_viz.msg import Trajectory
from geometry_msgs.msg import Pose, Pose, Point
from tf import transformations as trans
import tf


from vis_helpers import getHeader

def getMeshMarker(id, pnt):
    _model_marker = Marker()
    scale = 0.0002
    _model_marker.header.frame_id = "world"
    _model_marker.header.stamp = rospy.Time(0)
    _model_marker.ns = "dude"
    _model_marker.id = id
    _model_marker.action = Marker.ADD
    _model_marker.scale.x = scale
    _model_marker.scale.y = scale
    _model_marker.scale.z = scale
    _model_marker.color.a = 1.0
    _model_marker.color.r = 1.0
    _model_marker.color.g = 0.45
    _model_marker.color.b = 0.0

    _model_marker.pose.position.x = pnt[0]
    _model_marker.pose.position.y = pnt[1]
    _model_marker.pose.position.z = pnt[2]
    _q = trans.quaternion_from_euler(-pnt[3], 0., pnt[4]+np.pi/2)
    # _q = trans.quaternion_from_euler(0., 0., 0)

    # _model_marker.pose.orientation.x = _q[0]
    # _model_marker.pose.orientation.y = _q[1]
    # _model_marker.pose.orientation.z = _q[2]
    # _model_marker.pose.orientation.w = _q[3]
    _model_marker.pose.orientation.x = 0.
    _model_marker.pose.orientation.y = 0.
    _model_marker.pose.orientation.z = 0.
    _model_marker.pose.orientation.w = 0.

    # _model_marker.type = Marker.SPHERE
    _model_marker.type = Marker.MESH_RESOURCE
    _model_marker.mesh_resource = "package://time_optimal_ergodic_search/assets/drone.stl"
    return _model_marker

class AgentViz(object):

    def __init__(self, agent_name, scale=0.1):

        self._agent_name = agent_name

        self._scale = scale
        # instantiatiate the publishers
        self._model_pub     = rospy.Publisher(agent_name + '/vis' + '/model', MarkerArray, queue_size=10)
        self._traj_pub = rospy.Publisher(agent_name + '/vis' + '/traj', Marker, queue_size=10)

        self._model_msg = MarkerArray()
        # self._model_msg.ns = agent_name
        # self._model_msg.id = 0

        self._traj_msg = Marker()
        self._traj_msg.ns = agent_name
        self._traj_msg.id = 1
        # self._traj_sub = rospy.Subscriber(agent_name + '/planned_traj', Trajectory, self.callback_trajectory)


        self._tf_listener = tf.TransformListener()

        self.__build_rendering()

    def callback_trajectory(self, msg):
        # msg is of type Trajectory (see msg folder)
        # convert the points in the msg to a marker
        self._traj_msg.header = getHeader("world", rospy.Time(0))
        del self._traj_msg.points
        del self._model_msg.markers
        self._traj_msg.points = []
        self._model_msg.markers = []
        for pt in msg:
            self._traj_msg.points.append(Point(pt[0],pt[1],pt[2]))  
        stagged_pnts = msg[0::5]
        for i,pt in enumerate(stagged_pnts):
            self._model_msg.markers.append(getMeshMarker(i,pt))
        self._model_pub.publish(self._model_msg)
        self._traj_pub.publish(self._traj_msg)


    # def callback_trajectory(self, msg):
    #     # msg is of type Trajectory (see msg folder)
    #     # convert the points in the msg to a marker
    #     self._traj_msg.header = getHeader("world", rospy.Time(0))
    #     del self._traj_msg.points
    #     del self._model_msg.markers
    #     self._traj_msg.points = []
    #     self._model_msg.markers = []
    #     for pt in msg.points:
    #         self._traj_msg.points.append(pt)  
    #     stagged_pnts = [msg.points[0]] + msg.points[::5]    
    #     for i,pt in enumerate(stagged_pnts):
    #         self._model_msg.markers.append(getMeshMarker(i,pt))
    #     self._model_pub.publish(self._model_msg)
    #     self._traj_pub.publish(self._traj_msg)

    def update_model_pose(self):
            self._model_msg.header = getHeader("world", rospy.Time(0))
            (trans, rot) = self._tf_listener.lookupTransform(
                "world", self._agent_name, rospy.Time(0)
            )
            self._model_msg.pose.position.x = trans[0]
            self._model_msg.pose.position.y = trans[1]
            self._model_msg.pose.position.z = trans[2]
            self._model_msg.pose.orientation.x = rot[0]
            self._model_msg.pose.orientation.y = rot[1]
            self._model_msg.pose.orientation.z = rot[2]
            self._model_msg.pose.orientation.w = rot[3]
            self._model_pub.publish(self._model_msg)

    def run(self):
        while not rospy.is_shutdown():
            try: 
                self.update_model_pose()
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

    def __build_rendering(self):

        rgb = np.random.uniform(0,1, size=(3,))
        

        self._traj_msg.header.frame_id = "world"
        self._traj_msg.header.stamp = rospy.Time(0)
        self._traj_msg.ns = self._agent_name
        self._traj_msg.id = 1
        self._traj_msg.action = Marker.ADD
        self._traj_msg.points = []
        self._traj_msg.scale.x = 0.02
        self._traj_msg.color.a = 1.0
        self._traj_msg.color.r = 1.0
        self._traj_msg.color.g = 0.45
        self._traj_msg.color.b = 0.0
        self._traj_msg.type = Marker.LINE_STRIP


if __name__=="__main__":
    rospy.init_node('drone_viz')
    env = DroneViz(agent_name="dude")
    # env.run()
    rospy.spin()
