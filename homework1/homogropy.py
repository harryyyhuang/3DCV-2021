import sys
import argparse
import numpy as np
import cv2 as cv

from numpy.linalg import svd, norm, inv
WINDOW_NAME = 'window'

def parse_args():
    parser = argparse.ArgumentParser(description="Homography Estimation")
    parser.add_argument("Image1", type=str, default="images/1-0.png", help="Path to the first image")
    parser.add_argument("Image2", type=str, default="images/1-1.png", help="Path to the secod iamge")
    parser.add_argument("GroundTruth", type=str, help="Path to the ground truth numpy file")
    parser.add_argument("PointNumber", type=int, default=20, help="Number of points to calculate the homography")
    parser.add_argument("DistanceRatio", type=float, default=0.7, help="The ratio to perform distance ratio test")
    parser.add_argument("Demo", type=bool, default=True, help="If performing demo or not")

def getDistanceRatio(pointNum, image_Name):
    '''
    This function predefined the ratio used in ratio test
    For demo image 1-0 to 1-1: 0.09 for 4 points 0.105 for 8 points 0.128 for 20 points 0.2
    For demo image 1-0 to 1-2: 0.53 for 4 points 0.57 for 8 points 0.651 for 20 points
    '''
    if(image_Name == "images/1-1.png"):
        if(pointNum == 4):
            return 0.09
        elif(pointNum == 8):
            return 0.10
        elif(pointNum == 20):
            return 0.128
        else:
            return 0.2
    else:
        if(pointNum == 4):
            return 0.53
        elif(pointNum == 8):
            return 0.57
        elif(pointNum == 20):
            return 0.651
        else:
            return 0.2


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])  

def deleteOutlier(points1, points2, points_delete):
    '''
    This function is to delete the feature points that is selected to be deleted

    Input:
        points1: numpy array of features in first image
        points2: numpy array of features in second image
        points_delete: numpy array of features that is manually select

    Return:
        points1: numpy array [N, 2], N is the number of correspondences with removed points
        points2: numpy array [N, 2], N is the number of correspondences with removed points
    '''
    deleteIndex = []
    
    for k in range(len(points_delete)):
        for i in range(points1.shape[0]):
            if(abs(points1[i][0]-points_delete[k][0]) < 2):
                deleteIndex.append(i)

    for i in range(len(deleteIndex)):
        points1 = np.delete(points1, deleteIndex[i], axis=0)
        points2 = np.delete(points2, deleteIndex[i], axis=0)

    return points1, points2


def get_sift_correspondences(img1, img2, pointNum, image2_name, ratio, isDemo):
    '''
    This function create sift feature and calculate the k best matching points

    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image
        pointNum: number of correspondences to return
        image2_name: the name of image 2
        ratio: the ratio to use in ratio test, if is Demo we will use values that 
               we have received best accuracy
        isDemo: boolean check if is demoing

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''

    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # use predefined ratio if demoing
    if(isDemo):
        ratio = getDistanceRatio(pointNum, image2_name)

    # create the matcher
    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # store the features pass the ratio test
    good_matches = []                                   
    for m, n in matches:
        if m.distance < ratio * n.distance:                       
            good_matches.append(m)                             

    # sort all the feature points
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # convert the points into numpy
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    # visualize the matching points
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # manually select the outlier that is mismatched
    points_add= []
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])
    while True:
        img_draw_match_ = img_draw_match.copy()

        cv.imshow(WINDOW_NAME, img_draw_match_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exist when pressing ESC

    # delete the outlier that is selected
    points1, points2 = deleteOutlier(points1, points2, points_add)

    return points1, points2

def genNormalizedMatrix(points1, points2):
    '''
    This function is to calculate the similarity transform matrix

    Input:
        points1: unnormalized point features numpy array [N, 3]
        points2: unnormalized point features numpy array [N, 3]
    
    Return:
        T_1: similarity transform matrix of points1
        T_2: similarity transform matrix of points2
    '''
    points1_mean = np.mean(points1, axis=0)
    points2_mean = np.mean(points2, axis=0)

    points1_s_m1 = 1/np.sqrt(np.sum((points1-points1_mean)**2)/(2*points1.shape[0])) # fomula from  Wojciech Chojnacki, Michael J. Brooks, 
                                                                                # Anton van den Hengel, Darren Gawley, "Revisiting Hartley's Normalized Eight-Point Algorithm," 
                                                                                # IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 25, no. 9, pp. 1172-1177, Sept., 2003
    points2_s_m1 = 1/np.sqrt(np.sum((points2-points2_mean)**2)/(2*points2.shape[0])) # fomula from  Wojciech Chojnacki, Michael J. Brooks, 
                                                                                # Anton van den Hengel, Darren Gawley, "Revisiting Hartley's Normalized Eight-Point Algorithm," 
                                                                                # IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 25, no. 9, pp. 1172-1177, Sept., 2003
    T_1 = np.array([[points1_s_m1],[0],[-points1_s_m1*points1_mean[0]],
                    [0],[points1_s_m1],[-points1_s_m1*points1_mean[1]],
                    [0],[0],[1]])

    T_2 = np.array([[points2_s_m1],[0],[-points2_s_m1*points2_mean[0]],
                    [0],[points2_s_m1],[-points2_s_m1*points2_mean[1]],
                    [0],[0],[1]])

    return T_1.reshape((3,3)), T_2.reshape((3,3))

def normalizePoints(points1, points2):
    '''
    This function normalized the point value so that the average distance from the origin is square root of 2

    Input:
        points1: unnormalized point features numpy array [N, 3]
        points2: unnormalized point features numpy array [N, 3]
    
    Return:
        points1: normalized point features numpy array [N, 3]
        points2: normalized point features numpy array [N, 3]
    '''
    T_1, T_2 = genNormalizedMatrix(points1, points2)
    return points1@T_1.T, points2@T_2.T

def calculateError(points, gt_points):
    '''
    This function print the L1 Norm error between the transformed points and ground truth points 

    Input:
        points: points been transformed to gt_points frame, numpy array [N, 2]
        gt_points: ground truth points, numpy array [N, 2]
    

    '''
    print(norm(points-gt_points)/points.shape[0])

def transForm(points, H):
    '''
    This function transform the given points using homography matrix H

    Input:
        points: points to transform, numpy array [N, 2]
        H: numpy array [3, 3]
    
    Return:
        points: points been transformed, numpy array [N, 2]
    '''
    points = np.c_[points, np.ones(points.shape[0])]
    transPoints = points@H.T
    ratio = transPoints[:, -1]
    transPoints /= ratio[:, np.newaxis]
    return np.delete(transPoints, -1, axis=1)

def unNorm(H, points1, points2):
    '''
    This function unnormalized the point value which the average distance from the origin is square root of 2

    Input:
        points1: normalized point features numpy array [N, 3]
        points2: normalized point features numpy array [N, 3]
    
    Return:
        points1: unnormalized point features numpy array [N, 3]
        points2: unnormalized point features numpy array [N, 3]
    '''
    T_1, T_2 = genNormalizedMatrix(points1, points2)
    return inv(T_2)@H@T_1

def constructH(h):
    '''
    This function reshape the homography into matrix form 

    Input:
        h: numpy array [9]
    
    Return:
        h: numpy array [3, 3]
    '''
    return h.reshape((3,3))

def DLT(pMatrix):
    '''
    This function solve the homography by SVD method and homography matrix

    Input:
        pMatrix: numpy array [2*N, 9]
    
    Return:
        vh: numpy array [9]
    '''

    u, s, vh = svd(pMatrix)
    return vh[-1]

def toHomoForm(points1, points2):
    '''
    This function transform points into homogeneous form by adding 1 into last dimension

    Input:
        points1: numpy array [N, 2]
        points2: numpy array [N, 2]
    
    Return:
        points1: numpy array [N, 3]
        points2: numpy array [N, 3]
    '''
    return np.c_[points1, np.ones(points1.shape[0])], np.c_[points2, np.ones(points2.shape[0])]


def genPointMatrix(points1, points2):
    '''
    This function transform homogeneous point pair features into matrix form 

    Input:
        points1: numpy array [N, 3]
        points2: numpy array [N, 3]
    
    Return:
        matrix: numpy array [2*N, 9]
    '''
    matrix = np.zeros((2*points1.shape[0], 9))
    
    for i in range(points1.shape[0]):
        u_i = points1[i][0]
        v_i = points1[i][1]
        u_i_pla = points2[i][0]
        v_i_pla = points2[i][1]
        matrix[2*i] = np.array([u_i, v_i, 1, 0, 0, 0, -u_i_pla*u_i, -u_i_pla*v_i, -u_i_pla])
        matrix[2*i+1] = np.array([0, 0, 0, u_i, v_i, 1, -v_i_pla*u_i, -v_i_pla*v_i, -v_i_pla])


    return matrix

if __name__ == '__main__':

    # Parse the argument
    args = parse_args()

    # Read the files
    img1 = cv.imread(args.Image1)
    img2 = cv.imread(args.Image2)
    img2_name = args.Image2
    pointNum = args.PointNumber

    # Load the ground truth file if available
    if(args.GroundTruth):
        gt_correspondences = np.load(args.GroundTruth)
    
    # Matching the point features between two frames
    points1, points2 = get_sift_correspondences(img1, img2, pointNum, img2_name, args.DistanceRatio, args.Demo)

    # Transform the point features into Homogeneous form
    points1, points2 = toHomoForm(points1, points2)

    # Concate all the point features into Homography Matrix
    A = genPointMatrix(points1, points2)

    # Solve the homography by Direct Linear Transform
    h = DLT(A)

    # Reshape the homography into matrix form
    H = constructH(h)

    # Check the DLT error if groundtruth is given
    if(args.GroundTruth):
        gtpoints1, gtpoints2 = gt_correspondences[0], gt_correspondences[1]
        point2_trans = transForm(gtpoints1, H)
        print("The DLT error is")
        calculateError(point2_trans, gtpoints2)
    
    # Solve the homography by Normalized Direct Linear Transform
    points1_norm, points2_norm = normalizePoints(points1, points2)
    A_norm = genPointMatrix(points1_norm, points2_norm)
    h_norm = DLT(A_norm)
    H_norm = constructH(h_norm)
    H_NDLT = unNorm(H_norm, points1, points2)

    # Check the Normalized DLT error if groundtruth is given
    if(args.GroundTruth):
        point2_trans_norm = transForm(gtpoints1, H_NDLT)
        print("The normalized DLT error is")
        calculateError(point2_trans_norm, gtpoints2)

    
    
