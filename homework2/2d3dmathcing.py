from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import os.path
import time
from mypnpsolver import mypnpsolver
from display import displayPointcloud
import sys


images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
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

    return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)

def solvePnPOpencv():
        
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    # Load quaery image
    image_id = images_df["IMAGE_ID"].to_list()

    # store all the R and t
    store_R = []
    store_t = []
    gt_R    = []
    gt_t    = []

    if(os.path.isfile("Rotation.npy") and os.path.isfile("Translation.npy") and os.path.isfile("gtRotation.npy") and os.path.isfile("gtTranslation.npy") ):
        print("Rotation and Translation exist, skip solving PnP.")
        store_R = np.load("Rotation.npy") 
        store_t = np.load("Translation.npy")
        gt_R = np.load("gtRotation.npy") 
        gt_t = np.load("gtTranslation.npy")

    else:
        for i in range(len(image_id)):

            # process image one by one
            idx = image_id[i]
            print("processing {}/{} image {}".format(i, len(image_id), ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]))
            

            # read image
            fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
            rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

            # Load query keypoints and descriptors
            points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
            kp_query = np.array(points["XY"].to_list())
            desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
            ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
            rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
            tvec_gt = ground_truth[["TX","TY","TZ"]].values

            # Find correspondance and solve pnp
            retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query),(kp_model, desc_model))

            store_R.append(rvec)
            store_t.append(tvec)
            gt_R.append(rotq_gt)
            gt_t.append(tvec_gt)

        # store the result 
        np.save("Rotation", np.array(store_R))
        np.save("Translation", np.array(store_t))
        np.save("gtRotation", np.array(gt_R))
        np.save("gtTranslation", np.array(gt_t))
        

    # display the trajectory
    displayPointcloud((kp_model, np.array(desc_df['RGB'].to_list())), (store_R, store_t), True)

def getTranlationError(mineT, gtT):
    error = np.linalg.norm((mineT-gtT), axis=1)
    print("Median Translation Error is {}".format(np.median(error)))

def getRotationError(mineR, gtR):
    convertedR = R.from_matrix(mineR[:]).as_quat()
    invertR = R.from_quat(convertedR[:]).inv().as_quat()
    diff = gtR[:]*invertR[:]
    diff_as_axis = R.from_quat(diff[:]).as_rotvec()
    error = np.linalg.norm(diff_as_axis, axis=1)
    print("Median Rotation Error is {}".format(np.median(error)))



def solvePnPMine():
        
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    # Load quaery image
    image_id = images_df["IMAGE_ID"].to_list()

    # store all the R and t
    store_R = []
    store_t = []
    gt_R    = []
    gt_t    = []

    if(os.path.isfile("myRotation.npy") and os.path.isfile("myTranslation.npy") and os.path.isfile("gtRotation.npy") and os.path.isfile("gtTranslation.npy")):
        print("Rotation and Translation exist, skip solving PnP.")
        store_R = np.load("myRotation.npy") 
        store_t = np.load("myTranslation.npy")
        gt_R = np.load("gtRotation.npy") 
        gt_t = np.load("gtTranslation.npy")

    else:
        for i in range(len(image_id)):

            # process image one by one
            idx = image_id[i]
            print("processing {}/{} image {}".format(i, len(image_id), ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]))
            

            # read image
            fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
            rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

            # Load query keypoints and descriptors
            points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
            kp_query = np.array(points["XY"].to_list())
            desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
            ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
            rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
            tvec_gt = ground_truth[["TX","TY","TZ"]].values

            # Find correspondance and solve pnp
            retval, rvec, tvec, inliers = mypnpsolver((kp_query, desc_query),(kp_model, desc_model))

            store_R.append(rvec)
            store_t.append(tvec)
            gt_R.append(rotq_gt)
            gt_t.append(tvec_gt)

        store_R = np.array(store_R)
        store_t = np.array(store_t)
        gt_R = np.array(gt_R)
        gt_t = np.array(gt_t)
        # store the result 
        np.save("myRotation", store_R)
        np.save("myTranslation", store_t)
        np.save("gtRotation", np.array(gt_R))
        np.save("gtTranslation", np.array(gt_t))
        

    # display the trajectory
    displayPointcloud((kp_model, np.array(desc_df['RGB'].to_list())), (store_R, store_t), False)

    store_R = np.array(store_R)
    # calculate the Error
    for i in range(len(store_R)):
        store_t[i] = (-store_R[i]@(store_t[i].reshape(3,1))).reshape(3)
    gt_t = gt_t.reshape(gt_t.shape[0],gt_t.shape[2])
    gt_R = gt_R.reshape(gt_R.shape[0],gt_R.shape[2])
    getTranlationError(store_t, gt_t)
    getRotationError(store_R, gt_R)



if __name__ == '__main__':
    if(int(sys.argv[1]) == 1):
        solvePnPOpencv()
    else:
        solvePnPMine()
