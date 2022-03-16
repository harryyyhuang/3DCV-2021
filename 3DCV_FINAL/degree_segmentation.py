from numpy.lib.function_base import angle
from numpy.lib.histograms import histogram
from numpy.ma import ceil
import pykitti
import numpy as np
import cv2
from scipy.signal import savgol_filter
import queue
import open3d as o3d
import random
import open3d as o3d
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import joblib
import struct
from anomaly_onclass_SVM import pcd_processing, point_cloud_2_birdseye, point_cloud_2_birdseye_density
import argparse
pasrser = argparse.ArgumentParser()
pasrser.add_argument("pcd_path", help="path to the pcd file")
args = pasrser.parse_args()
def getR(points):
    tmpPoints = points.T
    R = np.sqrt(tmpPoints[0]**2 + tmpPoints[1]**2 + tmpPoints[2]**2)
    return R.T

def getPitch(points, R):
    pitches =  (points.T[2]/R)
    return np.arcsin(pitches)

def getYaw(points):
    return -np.arctan2(points.T[1], points.T[0])

def getU(pitches):
    fov_up = (2/180)*np.pi
    fov_down = (-24.8/180)*np.pi
    fov = fov_up-fov_down
    return np.round(64*(1-(pitches-fov_down)/fov))

def getV(yaws):
    return np.round(2000*(0.5*(yaws/np.pi+1)))

def drawDepth(img, u, v, R):
    max_R = np.max(R)
    min_R = np.min(R)

    for i in range(R.shape[0]):
        row = int(u[i])-1
        col = int(v[i])-1

        img[row][col] = ((R[i]-min_R)/(max_R-min_R))

    return img

def getLinear(img, i, j):
    value = 0
    count = 0
    if(i != 0 and img[i-1][j] != 0):
        value+=img[i-1][j]
        count+=1
    if(j != 0 and img[i][j-1] != 0):
        value+=img[i][j-1]
        count+=1
    if(i != 63 and img[i+1][j] != 0):
        value+=img[i+1][j]
        count+=1
    if(j != 1999 and img[i][j+1] != 0):
        value+=img[i][j+1]
        count+=1
    if(count == 0):
        value = np.max(img)
        count+=1
    return value/count

def billinear(img):
    for i in range(64):
        for j in range(2000):
            if(img[i][j] == 0):
                img[i][j] = getLinear(img, i, j)

    return img


def show_img(img):
    # img_show = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    img_max = np.max(img)
    img_min = np.min(img)

    img = 255*((img-img_min)/(img_max-img_min))
    img = img.astype(np.uint8)

    cv2.imshow("depth", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_angle(angle, img):
    del_rad = ((26.8/64)/180)*np.pi
    horizon = 2
    for j in range(2000):
        for i in range(64):
            if(i == 0):
                continue
            ti_a = abs(i-horizon)*del_rad
            ti_b = abs(i-1-horizon)*del_rad
            del_z = abs(img[i][j]*np.sin(ti_a)-img[i-1][j]*np.sin(ti_b))
            del_x = abs(img[i][j]*np.cos(ti_a)-img[i-1][j]*np.cos(ti_b))
            angle[i][j] = np.arctan2(del_z, del_x)

    return angle

def smooth(angle):
    angle_smooth = angle.T
    for i in range(angle_smooth.shape[0]):
        angle_smooth[i] = savgol_filter(angle_smooth[i], 11, 2)
    return angle_smooth.T

def getN4(r, c, ground):
    theN4 = []
    if(r != 0 and ground[r-1][c] == 0):
        theN4.append((r-1, c))
    if(c != 0 and ground[r][c-1] == 0):
        theN4.append((r, c-1))
    if(r != 63 and ground[r+1][c] == 0):
        theN4.append((r+1, c))
    if(c != 1999 and ground[r][c+1] == 0):
        theN4.append((r, c+1))
    return theN4
    
def labelGroundBFS(r, c, ground, angle):
    theQueue = queue.Queue(maxsize=ground.shape[0]*ground.shape[1])
    theQueue.put((r,c))
    while theQueue.empty() == False:
        ther, thec = theQueue.get()
        ground[ther][thec] = 1
        for r_n, c_n in getN4(ther, thec, ground):
            if(abs(angle[r_n][c_n]-angle[ther][thec]) < (5/180)*np.pi):
                ground[r_n][c_n] = 1
                theQueue.put((r_n, c_n))
    return ground
        

def groundLabel(ground, angle):
    for i in range(ground.shape[1]):
        if(angle[63][i] < (45/180)*np.pi):
            ground[63][i] = 1
    
    for i in range(ground.shape[1]):
        if(ground[62][i] == 0):
            ground = labelGroundBFS(62, i, ground, angle)

    return ground

def labelBFS(r, c, label_num, img, ground, label):
    theQueue = queue.Queue(maxsize=ground.shape[0]*ground.shape[1])
    theQueue.put((r,c))
    while theQueue.empty() == False:
        ther, thec = theQueue.get()
        if(ground[ther][thec] != 0):
            continue
        label[ther][thec] = label_num
        for r_n, c_n in getN4(ther, thec, label):
            d1 = max(img[r_n][c_n], img[ther][thec])
            d2 = min(img[r_n][c_n], img[ther][thec])
            if(r_n == ther):
                phi = ((360/2000)/180)*np.pi
            else:
                phi = ((26.8/64)/180)*np.pi
            up = d2*np.sin(phi)
            down = d1-d2*np.cos(phi)
            if(np.arctan2(up, down)>(10/180)*np.pi and ground[r_n][c_n] == 0):
                label[r_n][c_n] = label_num
                theQueue.put((r_n, c_n))

    return label

def label_image(img, ground, label):
    label_num = 2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(label[i][j]==0):
                label = labelBFS(i, j, label_num, img, ground, label)
                label_num+=1
    return label

colors = {}

def getColor(label):
    if(label not in colors):
        colors[label] = np.array([random.randint(0,255),random.randint(0,255),random.randint(0,255)])
    return colors[label]

def draw_color(u, v, ground, label, points):
    colors = np.zeros((u.shape[0],3))
    for i in range(u.shape[0]):
        row = int(u[i])-1
        col = int(v[i])-1
        # if(col>= 601 and col <= 1400):
        if(ground[row][col] == 1):
            colors[i] = [255,255,255]
        elif(label[row][col] != 0):
            colors[i] = getColor(label[row][col])
    return colors

def display_points(points, color):
    pcd = o3d.geometry.PointCloud()
    # points = np.delete(points, 3, 1)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color.astype(np.float)/255)

    displaysList = [pcd]

    o3d.visualization.draw_geometries(displaysList)

def cluster_points(u, v, label, points):
    cluster_points = {}
    for i in range(u.shape[0]):
        row = int(u[i])-1
        col = int(v[i])-1
        # if(col>= 601 and col <= 1400):
        if(label[row][col] != 0):
            theLabel = label[row][col]
            if(theLabel not in cluster_points):
                cluster_points[theLabel] = [points[i][:3].tolist()]
            else:
                cluster_points[theLabel].append(points[i][:3].tolist())
    return cluster_points

def getCenter(Points):
    return np.mean(Points.T, axis=1)
def getMaxHeight(Points):
    return np.max(Points.T, axis=1)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def process_block(thePoints):
    num = 0
    clf = OneClassSVM(gamma='scale', kernel='rbf', tol=1e-3, nu=0.05, verbose=False)
    clf = joblib.load('oneClassSVM_model.pkl')
    # print(clf)
    # exit()
    sel = VarianceThreshold(threshold=2.5)
    sel = joblib.load('feature_selection_model.pkl')
    for key in thePoints:
        pcd = o3d.geometry.PointCloud()
        if(len(thePoints[key]) < 100):
            continue
        tmpPoints = thePoints[key]
        print(key)
        pcd.points = o3d.utility.Vector3dVector(thePoints[key])
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.2)
        uni_down_pcd = pcd.uniform_down_sample(every_k_points=4)
        cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
        # display_inlier_outlier(voxel_down_pcd, ind)
        # displaysList = [cl]
        # o3d.visualization.draw_geometries(displaysList)
        val_set  = []
        allPoints = np.asarray(cl.points)
        allPoints = np.resize(allPoints, (1024, 3))
        # allPoints[:,[0, 1]] = allPoints[:,[1, 0]]
        # allPoints[:, 0] = - allPoints[:, 0]
        allPoints = pcd_processing(allPoints, "test")
        # print(allPoints)
        allPoints = point_cloud_2_birdseye(allPoints)
        allPoints = allPoints.flatten()
        val_set.append(allPoints)
        val_set = np.array(val_set)
        val_set = sel.transform(val_set)
        predict = clf.predict(val_set)
        print(predict)
        if(predict == 1):
            displaysList = [pcd]
            o3d.visualization.draw_geometries(displaysList)
            num+=1
    print(len(thePoints))
    print(num)

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)


if __name__ == '__main__':
    
    lidar_path = args.pcd_path

    points = read_bin_velodyne(lidar_path)

    R = getR(points)

    pitches = getPitch(points, R)

    yaws = getYaw(points)

    u = getU(pitches)

    v = getV(yaws)

    img = np.zeros((64,2000))

    img = drawDepth(img, u, v, R)

    img = billinear(img)

    angle_img = np.zeros((64, 2000))

    angle_img = get_angle(angle_img, img)

    ground = np.zeros((64, 2000))

    ground = groundLabel(ground, angle_img)

    label = np.zeros((64, 2000))

    label = label_image(img, ground, label)

    label_points = cluster_points(u, v, label, points)



    points_color = draw_color(u, v, ground, label, points)

    display_points(points, points_color)


