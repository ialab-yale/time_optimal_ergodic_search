import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Pose, Point
from tf import transformations as trans
import tf
import os
import sys


class EnvViz(object):
    def __init__(self, obs_tor, obs_box) -> None:

        self._env_msg = MarkerArray()
        self._msg_pub = rospy.Publisher('env_obs', MarkerArray, queue_size=10)
        self._id = 0
        for i,ob in enumerate(obs_tor):
            print("tor")
            _ob_marker = Marker()
            _ob_marker.header.frame_id = "world"
            _ob_marker.header.stamp = rospy.Time(0)
            _ob_marker.ns = "env"
            _ob_marker.id = self._id
            self._id += 1
            _ob_marker.action = Marker.ADD

            _ob_marker.pose.position.x = ob['pos'][0]
            _ob_marker.pose.position.y = ob['pos'][1]
            _ob_marker.pose.position.z = ob['pos'][2]
            _quat = trans.quaternion_about_axis(ob['rot'], (0,1,0))
            _ob_marker.pose.orientation.x = _quat[0]
            _ob_marker.pose.orientation.y = _quat[1]
            _ob_marker.pose.orientation.z = _quat[2]
            _ob_marker.pose.orientation.w = _quat[3]
            # _ob_marker.scale.x = ob['r2']/0.5
            # _ob_marker.scale.y = ob['r2']/0.5
            # _ob_marker.scale.z = ob['r2']/0.5
            _ob_marker.scale.x = 1.
            _ob_marker.scale.y = 1.
            _ob_marker.scale.z = 1.
            _ob_marker.color.a = 1.0
            rgb = np.random.uniform(0,1, size=(3,))
            _ob_marker.color.r = rgb[0]
            _ob_marker.color.g = rgb[1]
            _ob_marker.color.b = rgb[2]
            # _ob_marker.type = Marker.SPHERE
            _ob_marker.type = Marker.MESH_RESOURCE
            _ob_marker.mesh_resource = "package://time_optimal_ergodic_search/assets/sqorus_physical.stl"
            # _ob_marker.lifetime = 0 #<-- forever?
            self._env_msg.markers.append(_ob_marker)
        for i,ob in enumerate(obs_box):
            print("box")
            _ob_marker = Marker()
            _ob_marker.header.frame_id = "world"
            _ob_marker.header.stamp = rospy.Time(0)
            _ob_marker.ns = "env"
            _ob_marker.id = self._id
            self._id += 1
            _ob_marker.action = Marker.ADD

            _ob_marker.pose.position.x = ob['pos'][0]
            _ob_marker.pose.position.y = ob['pos'][1]
            _ob_marker.pose.position.z = ob['pos'][2]
            _quat = trans.quaternion_about_axis(ob['rot'], (0,0,1))
            _ob_marker.pose.orientation.x = _quat[0]
            _ob_marker.pose.orientation.y = _quat[1]
            _ob_marker.pose.orientation.z = _quat[2]
            _ob_marker.pose.orientation.w = _quat[3]
            _ob_marker.scale.x = ob['half_dims'][0]*2
            _ob_marker.scale.y = ob['half_dims'][1]*2
            _ob_marker.scale.z = ob['half_dims'][2]*2
            _ob_marker.color.a = 1.0
            rgb = np.random.uniform(0,1, size=(3,))
            _ob_marker.color.r = rgb[0]
            _ob_marker.color.g = rgb[1]
            _ob_marker.color.b = rgb[2]
            _ob_marker.type = Marker.CUBE
            # _ob_marker.lifetime = 0 #<-- forever?
            self._env_msg.markers.append(_ob_marker)

        self._msg_pub.publish(self._env_msg)

    def pub_env(self):
        self._msg_pub.publish(self._env_msg)

    def run(self):
        " keeps the node alive "
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            print('----- publishing markers -----')
            self._msg_pub.publish(self._env_msg)
            rate.sleep()