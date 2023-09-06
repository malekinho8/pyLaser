import sys; sys.path.append('./')
from utils import *

# define setup variables
laser_azimuth = 45
laser_polar = 155
mirror_roll = 0
mirror_pitch = -20
mirror_yaw = 0
intersection_plane_roll = 10
intersection_plane_pitch = 0
intersection_plane_yaw = 10
intersection_plane_point = np.array([10,-10,100])

# call main function
run(laser_azimuth, laser_polar, mirror_roll, mirror_pitch, mirror_yaw, intersection_plane_roll, intersection_plane_pitch, intersection_plane_yaw, intersection_plane_point)
