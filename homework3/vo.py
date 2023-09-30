from io import SEEK_CUR

from numpy.core.records import array
import open3d as o3d
import numpy as np
import cv2
import sys, os, argparse, glob
import multiprocessing as mp
# from scipy.spatial.transform import Rotation as R
import random
from visual_odometry import VisualOdometry, PinholeCamera

class SimpleVO:
    def __init__(self, args):
        
        self.frame_paths = args.input
        self._vid = None
        self.K = np.array([[744.5709868658486, 0.0, 627.1323814675709],
                            [0.0, 745.8059206233886, 602.7107161940445],
                            [0.0, 0.0, 1.0]])
        self.dist = np.array(([[-0.20070319857803537, 0.024192241409856645, -0.00014815608365824698, -0.0005627410085481332, -0.0002250428307017009]]))

        self._R_prev = np.eye(3, 3)
        self._t_prev = np.zeros((3, 1))

    def run(self):
        self.process_frames()

    def process_frames(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        points = np.array([[0, 0, 1], [1200/2, 0, 1], [1200/2, 1328/2, 1], [0, 1328/2, 1]])
        points = np.linalg.pinv(self.K)@points.T
        points =  self._t_prev + self._R_prev@points
        points = points.T 

        # get center and concate all
        all_points = np.ones((points.shape[0]+1, points.shape[1]))
        all_points[:-1, :] = points

        all_points[-1, :] = self._t_prev.T

        # set up line
        line_set = o3d.geometry.LineSet(
            points = o3d.utility.Vector3dVector(all_points),
            lines = o3d.utility.Vector2iVector([[0,1], [1,2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
        )

        # generate color
        color = [0, 0, 0]
        colors = np.tile(color, (8,1))

        line_set.colors = o3d.utility.Vector3dVector(colors)
        self._prev_line_set = line_set
        vis.add_geometry(line_set)
        coord_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0.0, 0.0, 0.0])) # red x green y blue z
        vis.add_geometry(coord_obj)
        self._current_pos = self._t_prev

        self._vid = cv2.VideoCapture(self.frame_paths)
        cam = PinholeCamera(1328, 1200,
                            744.5709868658486, 745.8059206233886, 627.1323814675709, 602.7107161940445
                            -0.20070319857803537, 0.024192241409856645, -0.00014815608365824698, -0.0005627410085481332, -0.0002250428307017009)
        vo = VisualOdometry(cam)

        count = 0
        while(self._vid.isOpened()):
            ret, img = self._vid.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            vo.update(img)

            count += 1

            # queue.put((vo.cur_R, vo.cur_t))
            if (vo.cal_R is None):
                current_R = self._R_prev
            else:
                current_R = vo.cal_R
            # print(current_R)
            self._prev_line_set = self._prev_line_set.rotate(current_R, center=(0, 0, 0))
            vis.update_geometry(self._prev_line_set)
             
            cv2.imshow('frame', img)
            img_query = img
            if cv2.waitKey(30) == 27: break

            vis.poll_events()
            vis.update_renderer()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
