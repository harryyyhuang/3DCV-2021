from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
from numpy.linalg import inv, norm, eig, pinv, det
import sys
import cv2
import time
from argparse import Namespace
import random


def undistortion(v, invCameraMatrix, distCoeffs):
    centerv = (invCameraMatrix@(np.array([0.0,0.0,1.0]).reshape(3,1)))

    # Brown-Conrady Model
    r = np.sqrt((v[:, 0] - centerv[0])**2 + (v[:, 1] - centerv[1])**2) 
    v_u_x = v[:, 0] + \
            (v[:, 0] - centerv[0])*(distCoeffs[0]*(r**2)+distCoeffs[1]*(r**4)) + \
            (distCoeffs[2] * (r**2 + 2*(v[:, 0] - centerv[0])**2) + 2 * distCoeffs[3] * (v[:, 0] - centerv[0]) * (v[:, 1] - centerv[1]))

    v_u_y = v[:, 1] + \
            (v[:, 1] - centerv[1])*(distCoeffs[0]*(r**2)+distCoeffs[1]*(r**4)) + \
            (2 * distCoeffs[2] * (v[:, 0] - centerv[0]) * (v[:, 1] - centerv[1]) + distCoeffs[3] * (r**2 + 2*(v[:, 1] - centerv[1])**2))
    

    new_v = np.stack((v_u_x, v_u_y, v[:, 2]))

    return new_v.T

def getv(point2D, cameraMatrix, distCoeffs):
    h_point2D = np.ones((point2D.shape[0], point2D.shape[1]+1))
    h_point2D[:,:-1] = point2D

    # inverse H
    invCameraMatrix = inv(cameraMatrix)
    v = (invCameraMatrix@h_point2D.T).T


    centerv = (invCameraMatrix@(np.array([0.0,0.0,1.0]).reshape(3,1)))

    # Brown-Conrady Model
    r = np.sqrt((v[:, 0] - centerv[0])**2 + (v[:, 1] - centerv[1])**2) 
    v_u_x = v[:, 0] + \
            (v[:, 0] - centerv[0])*(distCoeffs[0]*(r**2)+distCoeffs[1]*(r**4)) + \
            (distCoeffs[2] * (r**2 + 2*(v[:, 0] - centerv[0])**2) + 2 * distCoeffs[3] * (v[:, 0] - centerv[0]) * (v[:, 1] - centerv[1]))

    v_u_y = v[:, 1] + \
            (v[:, 1] - centerv[1])*(distCoeffs[0]*(r**2)+distCoeffs[1]*(r**4)) + \
            (2 * distCoeffs[2] * (v[:, 0] - centerv[0]) * (v[:, 1] - centerv[1]) + distCoeffs[3] * (r**2 + 2*(v[:, 1] - centerv[1])**2))
    

    new_v = np.stack((v_u_x, v_u_y, v[:, 2]))
    
    return v
    

def solvetrilatiration(p1, p2, p3, a, b, c):

    all_center = []

    for i in range(len(a)):
        v_1 = (p2 - p1) / (norm(p2 - p1))
        v_2 = (p3 - p1) / (norm(p3 - p1))

        ix = v_1
        iz = np.cross(v_1, v_2) / norm(np.cross(v_1, v_2))
        iy = np.cross(ix, iz) / norm(np.cross(ix, iz))

        x2 = norm(p2 - p1)
        x3 = (p3-p1)@ix
        y3 = (p3-p1)@iy
        
        x_length = (a[i]**2 - b[i]**2 + x2**2)/(2*x2)


        y_length = (a[i]**2 - c[i]**2 + x3**2 + y3**2 - (2*x3*x_length))/(2*y3)
        

        z_length = np.sqrt(a[i]**2 - x_length**2 - y_length**2)

        direction = x_length*ix + y_length*iy + z_length*iz
        direction_minus = x_length*ix + y_length*iy - z_length*iz

        all_center.append(p1+direction_minus)
        all_center.append(p1+direction)

    return all_center


def calculateError(v, point):
    return norm(v-point)

def reproject(v_4, point3D_4, all_R, all_T, cameraMatrix):

    lowesrError = sys.float_info.max
    finalR = all_R[0]
    finalT = all_T[0]

    for i in range(len(all_R)):
        point = (all_R[i]@(point3D_4.reshape(3,1)-all_T[i].reshape(3,1)))
        point = point / point[2]
        point = (cameraMatrix@(all_R[i]@(point3D_4.reshape(3,1)-all_T[i].reshape(3,1)))).T
        point = (point / point[0][2])[0][:2]
        error = calculateError(v_4, point)
        if(error < lowesrError):
            finalR = all_R[i]
            finalT = all_T[i]
            lowesrError = error

    return finalR, finalT


def mysolveP3P(point3D, point2D, cameraMatrix, distCoeffs):

    point2D_4 = point2D[3]
    point3D_4 = point3D[3]

    # homogeneous 2d points
    v = getv(point2D, cameraMatrix, distCoeffs)
    v = v[:3]
    point3D = point3D[:3]



    # create given

    C_ab = (v[0]@v[1])/(norm(v[0])*norm(v[1]))
    C_bc = (v[1]@v[2])/(norm(v[1])*norm(v[2]))
    C_ac = (v[0]@v[2])/(norm(v[0])*norm(v[2]))



    R_ab = norm(point3D[0]-point3D[1])
    R_bc = norm(point3D[1]-point3D[2])
    R_ac = norm(point3D[0]-point3D[2])

    K_1 = (R_bc/R_ac)**2
    K_2 = (R_bc/R_ab)**2

    G_4 = (K_1*K_2 - K_1 - K_2)**2 - 4*K_1*K_2*(C_bc**2)
    G_3 = 4*(K_1*K_2 - K_1 - K_2)*K_2*(1 - K_1)*C_ab \
        + 4*K_1*C_bc*((K_1*K_2 - K_1 + K_2) * C_ac + 2*K_2*C_ab*C_bc)
    G_2 = (2*K_2*(1-K_1)*C_ab)**2 \
        + 2*(K_1*K_2 - K_1 - K_2)*(K_1*K_2 + K_1 - K_2) \
        + 4*K_1*((K_1 - K_2)*(C_bc**2) + K_1*(1-K_2)*(C_ac**2) - 2*(1+K_1)*K_2*C_ab*C_ac*C_bc)
    G_1 = 4*(K_1*K_2 + K_1 - K_2)*K_2*(1-K_1)*C_ab \
        + 4*K_1*((K_1*K_2 - K_1 + K_2)*C_ac*C_bc + 2*K_1*K_2*C_ab*(C_ac**2))
    G_0 = (K_1*K_2 + K_1 - K_2)**2 \
        - 4*(K_1**2)*K_2*(C_ac**2)

    # find root of polynomial of x
    param = np.array([G_4, G_3, G_2, G_1, G_0])
    root = np.roots(param)
    real_root = np.array([num for num in root if np.isreal(num)]).astype(float)    
    
    # compute a compute y 
    a = np.sqrt((R_ab**2)/(1+(real_root**2)-2*real_root*C_ab))
    m = (1 - K_1)
    p = 2*(K_1 * C_ac - real_root*C_bc)
    q = (real_root**2 - K_1)
    m_pla = 1
    p_pla = 2*(-real_root*C_bc)
    q_pla = (real_root**2)*(1-K_2) + 2*real_root*K_2*C_ab - K_2
    y = -(m_pla*q - m*q_pla)/(p*m_pla - p_pla*m)
    b = real_root*a
    c = y*a

    center = solvetrilatiration(point3D[0], point3D[1], point3D[2],
                  a, b, c)

    all_R = []
    all_T = []

    for i in range(len(center)):
        for j in range(2):
            lada = norm((point3D-center[i]), axis=1)/norm(v, axis=1)
            lada_sign = ((-1)**j)
            lada = lada*lada_sign
            R = (v.T*lada)@pinv(point3D.T-center[i].reshape(3,1))
            all_R.append(R)
            all_T.append(center[i])


    bestR, bestT = reproject(point2D_4, point3D_4, all_R, all_T, cameraMatrix)

    return bestR, bestT


def checkinLier(point3D, point2D, R, t, cameraMatrix, distCoeffs, param):

    point = (cameraMatrix@(R@(point3D.T-t.reshape(3,1))))
    point = (point / point[2])
    point = point[:2].T

    error = norm((point2D - point), axis = 1)

    return np.where(error < param.d)[0].shape[0], np.where(error < param.d)[0]
    


def mysolvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs):

    # take 72 points e = 0.4 p = 0.99 s = 4
    param = Namespace(
        s = 4,
        e = 0.5,
        p = 0.99,
        d = 10,
    )

    origin_total = points3D.shape[0]
    num_iter = (np.log(1-param.p))/(np.log(1-(1-param.e)**param.s))
    terminate = (1-param.e)*points3D.shape[0]

    maxinlier = 0
    minR = 0
    minT = 0
    num = 0
    while(True):
        print("starting {} turns in total {}".format(num, int(num_iter)))
        solvePoints = np.random.randint(points3D.shape[0], size=4)
        firstThreePoints3D = points3D[solvePoints]
        firstThreePoints2D = points2D[solvePoints]

        try:
            R, t = mysolveP3P(firstThreePoints3D, firstThreePoints2D, cameraMatrix, distCoeffs)

            inlierNum, inlier = checkinLier(points3D, points2D, R, t, cameraMatrix, distCoeffs, param)
            if (inlierNum >= int(origin_total*param.e)):
                maxinlier = inlierNum
                points3D = points3D[inlier]
                points2D = points2D[inlier]
                minR = R
                minT = t

        except:
            print("fail")

        num+=1
            

        if(num >= num_iter):
            print("max inlierNum {}".format(maxinlier))
            break


    return True, minR, minT, "grabage"

def mypnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query, desc_model, k=2)

    gmatches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    return mysolvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)