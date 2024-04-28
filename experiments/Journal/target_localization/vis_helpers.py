import numpy as np 
import rospy 
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header


def getHeader(frame_id, time):
    header = Header()
    header.frame_id = frame_id
    header.stamp = time 
    return header

def getColor(color):
    return None

