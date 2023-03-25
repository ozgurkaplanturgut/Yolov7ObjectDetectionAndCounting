from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np

def check_zone(x,y,zone):
    centroid = Point(x,y)
    polygon = Polygon(zone)
    if polygon.contains(centroid):
        return True
    else:
        return False

def create_zone(frame,area):
    area = np.array(area,np.int32)
    area.reshape(-1,1,2)
    cv2.polylines(frame,[area],True,(255,0,255),2)

    #return frame

