#!/usr/bin/env python3

import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from drone_env_viz.msg import Trajectory
from geometry_msgs.msg import Pose, Pose, Point
from tf import transformations as trans
import tf


def getMeshMarker(id, args):
    _model_marker = Marker()
    scale = 1
    _model_marker.header.frame_id = "world"
    _model_marker.header.stamp = rospy.Time(0)
    _model_marker.ns = "dude"
    _model_marker.id = id
    _model_marker.action = Marker.ADD
    _model_marker.scale.x = scale
    _model_marker.scale.y = scale
    _model_marker.scale.z = scale
    _model_marker.color.a = 1.0
    _model_marker.color.r = 0.0
    _model_marker.color.g = 1.0
    _model_marker.color.b = 0.0

    _model_marker.pose.position.x = args['pos'][0]
    _model_marker.pose.position.y = args['pos'][1]
    _model_marker.pose.position.z = 1
    _model_marker.pose.orientation.x = 0
    _model_marker.pose.orientation.y = 0
    _model_marker.pose.orientation.z = 0
    _model_marker.pose.orientation.w = 1
    _model_marker.scale.x = 1.0
    _model_marker.scale.y = 1.0
    _model_marker.scale.z = 1.0 
    _model_marker.color.a = 1.0

    _model_marker.type = Marker.MESH_RESOURCE
    _model_marker.mesh_resource = "package://time_optimal_ergodic_search/assets/CartoonTree.stl"
    return _model_marker

class EnvViz(object):
    def __init__(self, obs) -> None:
        self._env_msg = MarkerArray()

        self._msg_pub = rospy.Publisher('env_obs', MarkerArray, queue_size=10)

        print('----- building markers -----')
        # obs = rospy.get_param('obstacles')
        for i,ob in enumerate(obs):
            marker = getMeshMarker(i, ob)
            self._env_msg.markers.append(marker)
    
    def publish_msg(self):
        self._msg_pub.publish(self._env_msg)
