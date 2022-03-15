from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import time
from mypnpsolver import mypnpsolver
from display import displayPointcloud
import re
import sys
import os.path

class Point_class():
    def __init__(self, position, color):
        self.position = position
        self.color = color

def generate_points(cube_vertice):

    point_list = []
    # top
    top_surface = list(cube_vertice[:4])
    top_x = (top_surface[1]-top_surface[0])/9
    top_y = (top_surface[2]-top_surface[0])/9
    for i in range(8):
        point_row_pose = top_surface[0] + (i+1)*top_x
        for j in range(10):
            point_pose = point_row_pose + j*top_y
            point_list.append(Point_class(point_pose, (255, 0, 0)))

    # front
    front_surface = list(cube_vertice[[0,1,4,5]])
    front_x = (front_surface[1]-front_surface[0])/9
    front_y = (front_surface[2]-front_surface[0])/9
    for i in range(8):
        point_row_pose = front_surface[0] + (i+1)*front_x
        for j in range(8):
            point_pose = point_row_pose + (j+1)*front_y
            point_list.append(Point_class(point_pose, (0, 255, 0)))

    # back
    back_surface = list(cube_vertice[[2, 3, 6, 7]])
    back_x = (back_surface[1]-back_surface[0])/9
    back_y = (back_surface[2]-back_surface[0])/9
    for i in range(8):
        point_row_pose = back_surface[0] + (i+1)*back_x
        for j in range(8):
            point_pose = point_row_pose + (j+1)*back_y
            point_list.append(Point_class(point_pose, (255, 0, 255)))


    # botton
    botton_surface = list(cube_vertice[[4, 5, 6, 7]])
    botton_x = (botton_surface[1]-botton_surface[0])/9
    botton_y = (botton_surface[2]-botton_surface[0])/9
    for i in range(8):
        point_row_pose = botton_surface[0] + (i+1)*botton_x
        for j in range(10):
            point_pose = point_row_pose + j*botton_y
            point_list.append(Point_class(point_pose, (0, 0, 255)))

    # right
    right_surface = list(cube_vertice[[1, 3, 5, 7]])
    right_x = (right_surface[1]-right_surface[0])/9
    right_y = (right_surface[2]-right_surface[0])/9
    for i in range(10):
        point_row_pose = right_surface[0] + i*right_x
        for j in range(10):
            point_pose = point_row_pose + j*right_y
            point_list.append(Point_class(point_pose, (255, 255, 0)))

    # left
    left_surface = list(cube_vertice[[0, 2, 4, 6]])
    left_x = (left_surface[1]-left_surface[0])/9
    left_y = (left_surface[2]-left_surface[0])/9
    for i in range(10):
        point_row_pose = left_surface[0] + i*left_x
        for j in range(10):
            point_pose = point_row_pose + j*left_y
            point_list.append(Point_class(point_pose, (0, 255, 255)))



    return point_list



def draw_cube_opencv(img, rotation, translation, cube_Rt, cube_vertice):
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    

    rotation = R.from_quat(rotation).as_matrix()[0]

    camera_pos = (-np.linalg.pinv(rotation)@translation.T).T

    # process top surface first
    # generate point
    points = generate_points(cube_vertice)
    
    points.sort(key= lambda point : np.linalg.norm((point.position-camera_pos)), reverse=True)
    for i in range(len(points)):
        pixel = (cameraMatrix@(rotation@(points[i].position-camera_pos).T)).T
        pixel = (pixel/pixel[0][2])[0]
        if((pixel<0).any()):
            continue
        img = cv2.circle(img, (int(pixel[0]), int(pixel[1])), radius=5, color=points[i].color, thickness=-1)

    return img

def draw_cube(img, rotation, translation, cube_Rt, cube_vertice):
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    

    points = generate_points(cube_vertice)
    
    points.sort(key= lambda point : np.linalg.norm((point.position-translation)), reverse=True)
    for i in range(len(points)):
        pixel = (cameraMatrix@(rotation@(points[i].position-translation).T)).T
        pixel = (pixel/pixel[2])
        if((pixel<0).any()):
            continue
        img = cv2.circle(img, (int(pixel[0]), int(pixel[1])), radius=5, color=points[i].color, thickness=-1)
        

    return img

def get_valid_name(images_df):

    image_name = images_df["IMAGE_ID"].to_list()
    valid_img_name = []
    # collect name
    for i in range(len(image_name)):
        idx = image_name[i]

        if("valid" in ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]):
            print("processing {}/{} image {}".format(i, len(image_name), ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]))

        else:
            continue


        fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        valid_img_name.append(fname)

    valid_img_name = sorted(valid_img_name, key = lambda name: int(name[name.find('g')+1:name.find('.')]))

    return valid_img_name

def get_image(valid_img_name):
    valid_img = []
    # read image
    for i in range(len(valid_img_name)):
        rimg = cv2.imread("data/frames/"+valid_img_name[i])
        valid_img.append(rimg)
    return valid_img


def videoByMine():

    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # sort it
    valid_img_name = get_valid_name(images_df)
    valid_img = get_image(valid_img_name)
    shape = (int(valid_img[0].shape[1]), int(valid_img[0].shape[0]))

    store_R = []
    store_t = []
    load_R = []
    load_t = []
    if(os.path.isfile("myRotation.npy") and os.path.isfile("myTranslation.npy") ):
        print("Not generate extrinsic yet !!! Please Run the \"python3 2d3dmathcing.py 2\" first!!!")
        load_R = np.load("myRotation.npy") 
        load_t = np.load("myTranslation.npy")


    for i in range(len(valid_img_name)):

        print("processing {}/{} image {}".format(i, len(valid_img_name), valid_img_name[i]))

        idx = ((images_df.loc[images_df["NAME"] == valid_img_name[i]])["IMAGE_ID"].values)[0]

        fname = valid_img_name[i]
        rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

        store_R.append(load_R[idx-1])
        store_t.append(load_t[idx-1])

    # draw cube
    cube_Rt = np.load("cube_transform_mat.npy")
    cube_vertice = np.load("cube_vertices.npy")

    for i in range(len(valid_img)):
        valid_img[i] = draw_cube(valid_img[i], store_R[i], store_t[i], cube_Rt, cube_vertice)

    out = cv2.VideoWriter("ARVideo.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 15, shape)

    for i in range(len(valid_img)):
        out.write(valid_img[i])

    cv2.destroyAllWindows()
    out.release()



def videoByOpenCV():

    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # sort it
    valid_img_name = get_valid_name(images_df)
    valid_img = get_image(valid_img_name)
    shape = (int(valid_img[0].shape[1]), int(valid_img[0].shape[0]))

    store_R = []
    store_t = []
    load_R = []
    load_t = []
    if(os.path.isfile("Rotation.npy") and os.path.isfile("Translation.npy") ):
        print("Not generate extrinsic yet !!! Please Run the \"python3 2d3dmathcing.py 1\" first!!!")
        load_R = np.load("Rotation.npy") 
        load_t = np.load("Translation.npy")


    for i in range(len(valid_img_name)):

        print("processing {}/{} image {}".format(i, len(valid_img_name), valid_img_name[i]))

        idx = ((images_df.loc[images_df["NAME"] == valid_img_name[i]])["IMAGE_ID"].values)[0]

        fname = valid_img_name[i]
        rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

        store_R.append(load_R[idx-1])
        store_t.append(load_t[idx-1])

    # draw cube
    cube_Rt = np.load("cube_transform_mat.npy")
    cube_vertice = np.load("cube_vertices.npy")

    for i in range(len(valid_img)):
        valid_img[i] = draw_cube_opencv(valid_img[i], store_R[i], store_t[i], cube_Rt, cube_vertice)

    out = cv2.VideoWriter("ARVideoOpen.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 15, shape)

    for i in range(len(valid_img)):
        out.write(valid_img[i])

    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    if(int(sys.argv[1]) == 1):
        videoByOpenCV()
    else:
        videoByMine()