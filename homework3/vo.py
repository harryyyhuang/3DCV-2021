from io import SEEK_CUR

from numpy.core.records import array
import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
# from scipy.spatial.transform import Rotation as R

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    #TODO:
                    # insert new camera pose here using vis.add_geometry()
                        # get four corner points and camera
                        # get four corner points and camera 
                    # get four corner points and camera 
                    points = np.array([[0, 0, 1], [360, 0, 1], [360, 640, 1], [0, 640, 1]])
                    points = np.linalg.pinv(self.K)@points.T
                    points =  t + np.linalg.pinv(R)@points
                    points = points.T 

                    # get center and concate all
                    all_points = np.ones((points.shape[0]+1, points.shape[1]))
                    all_points[:-1, :] = points

                    all_points[-1, :] = t.T


                    # set up line
                    line_set = o3d.geometry.LineSet(
                        points = o3d.utility.Vector3dVector(all_points),
                        lines = o3d.utility.Vector2iVector([[0,1], [1,2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
                    )

                    # generate color
                    color = [0, 0, 0]
                    colors = np.tile(color, (8,1))

                    line_set.colors = o3d.utility.Vector3dVector(colors)

                    vis.add_geometry(line_set)
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def process_frames(self, queue):
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        img_query = cv.imread(self.frame_paths[0])
        first_time = True
        for frame_path in self.frame_paths[1:]:
            img = cv.imread(frame_path)
            #TODO: compute camera pose here

            # Init ORB
            orb = cv.ORB_create()

            # find keypoints and descriptors with ORB
            if(first_time == True):
                kp_first, des_first = orb.detectAndCompute(img_query, None)
                kp_second, des_second = orb.detectAndCompute(img, None)

                # create BFMatcher object
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

                # Match descriptors.
                matches = bf.match(des_first, des_second)

                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)

                # to pixels
                points_query = np.empty((0,2))
                points_img = np.empty((0,2))
                past_indexis = []
                # self.des_past = []
                for mat in matches:
                    query_index = mat.queryIdx
                    img_index = mat.trainIdx
                    # self.des_past.append(des_second[img_index])
                    past_indexis.append(img_index)
                    points_query = np.vstack((points_query, kp_first[query_index].pt))
                    points_img = np.vstack((points_img, kp_second[img_index].pt))

                # self.des_past = des_second[this_indexis]
                # self.kp_past = np.array(kp_second)[this_indexis]
                self.des_past = des_second[past_indexis]
            
                norm_query = cv.undistortPoints(points_query, self.K, self.dist, None, self.K)
                norm_img = cv.undistortPoints(points_img, self.K, self.dist, None, self.K)
           
                E, inlier = cv.findEssentialMat(norm_query, norm_img, self.K)

                points, R_this, t_this, inlier = cv.recoverPose(E, norm_query, norm_img, self.K)

                inlier_index = np.squeeze(np.argwhere(np.squeeze(inlier) == 255))
                norm_query_inlier =  norm_query[inlier_index]
                norm_img_inlier =  norm_img[inlier_index]
                self.des_past = self.des_past[inlier_index]

                # R_old = R
                # t_old = t
                self.R_old = R
                self.t_old = -t
                R = R_this
                t = -t_this
                # print(norm_query.shape)
                self.X_last = cv.triangulatePoints(np.hstack((self.R_old,self.t_old)), np.hstack((R, t)), norm_query_inlier, norm_img_inlier)
                self.X_last = (self.X_last/self.X_last[-1]).T
                # self.img1 = img_query
                # self.img2 = img

                first_time = False

            else:
                # kp_img1, des_img1 = orb.detectAndCompute(self.img1, None)
                # kp_img2, des_img2 = orb.detectAndCompute(self.img2, None)
                # kp_img3, des_img3 = orb.detectAndCompute(img, None)
                kp_first, des_first = orb.detectAndCompute(img_query, None)
                kp_second, des_second = orb.detectAndCompute(img, None)

                # create BFMatcher object
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

                # Match descriptors.
                matches = bf.match(des_first, des_second)

                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)

                 # to pixels
                points_query = np.empty((0,2))
                points_img = np.empty((0,2))
                past_indexis = []
                # self.des_past = []
                for mat in matches:
                    query_index = mat.queryIdx
                    img_index = mat.trainIdx
                    # self.des_past.append(des_second[img_index])
                    past_indexis.append(img_index)
                    points_query = np.vstack((points_query, kp_first[query_index].pt))
                    points_img = np.vstack((points_img, kp_second[img_index].pt))

                # self.des_past = des_second[this_indexis]
                # self.kp_past = np.array(kp_second)[this_indexis]
                des_second = des_second[past_indexis]

                # matches = bf.match(des_img2, des_img3)

                # matches = sorted(matches, key = lambda x:x.distance)

                norm_img1 = cv.undistortPoints(points_query, self.K, self.dist, None, self.K)
                norm_img2 = cv.undistortPoints(points_img, self.K, self.dist, None, self.K)
        
                E, _ = cv.findEssentialMat(norm_img1, norm_img2, self.K)
                _, R_this, t_this, inlier = cv.recoverPose(E, norm_img1, norm_img2, self.K)
                

                # inlier_index_12 = np.squeeze(np.argwhere(np.squeeze(inlier_12) == 0))
                # norm_img1 =  norm_img1[inlier_index_12]
                # norm_img2 =  norm_img2[inlier_index_12]
                # norm_img3 =  norm_img3[inlier_index_12]
                # E_23, _ = cv.findEssentialMat(norm_img2, norm_img3, self.K)
                # _, R_23, t_23, inlier_23 = cv.recoverPose(E_23, norm_img2, norm_img3, self.K)
                # inlier_index_23 = np.squeeze(np.argwhere(np.squeeze(inlier_23) == 0))
                # norm_img1 =  norm_img1[inlier_index_23]
                # norm_img2 =  norm_img2[inlier_index_23]
                # norm_img3 =  norm_img3[inlier_index_23]
                # Match descriptors.
                matches = bf.match(self.des_past, des_second)

                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)

                # to pixels
                points_3D_query = np.empty((0,4))
                points_img_first = np.empty((0, 2))
                points_img_second = np.empty((0, 2))
                for mat in matches:
                    query_index = mat.queryIdx
                    img_index = mat.trainIdx
                    points_3D_query = np.vstack((points_3D_query, self.X_last[query_index]))
                    points_img_first = np.vstack((points_img_first, norm_img1[img_index]))
                    points_img_second = np.vstack((points_img_second, norm_img2[img_index]))

                points_img_first = points_img_first.reshape(-1,1,2)
                points_img_second = points_img_second.reshape(-1,1,2)
                

                inlier_index = np.squeeze(np.argwhere(np.squeeze(inlier) == 255))
                norm_query_inlier =  norm_img1[inlier_index]
                norm_img_inlier =  norm_img2[inlier_index]
                des_second = des_second[inlier_index]
                
                t_this = -t_this
                t_tmp = t + (R@t_this)
                R_tmp = R@R_this
                # X_old = cv.triangulatePoints(np.hstack((self.R_old,self.t_old)), np.hstack((R, t)), norm_img1, norm_img2)
                X = cv.triangulatePoints(np.hstack((R,t)), np.hstack((R_tmp, t_tmp)), points_img_first, points_img_second)
                X = X/X[-1]
                points_3D_query = points_3D_query.T
                # # assert(X_old.shape == X.shape)

                # X_old = X_old/X_old[-1]
                

                ratio = np.linalg.norm(X[:, 0]- X[:, 1])/np.linalg.norm(points_3D_query[:, 0]- points_3D_query[:, 1]) 
                # print(np.linalg.norm(X[:, 0]- X[:, 1]))
                if(ratio > 2):
                    ratio = 2

                
                # self.R_old = R
                # self.t_old = t

                t_tmp = t + ratio*(R@t_this)
                R_tmp = R@R_this
                self.des_past = des_second
                self.X_last = cv.triangulatePoints(np.hstack((R,t)), np.hstack((R_tmp, t_tmp)), norm_query_inlier, norm_img_inlier)
                self.X_last = (self.X_last/self.X_last[-1]).T
                R = R_tmp
                t = t_tmp

            queue.put((R, t))
             
            cv.imshow('frame', img)
            img_query = img
            if cv.waitKey(30) == 27: break
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
