import sys
import argparse
import numpy as np
import cv2 as cv

from numpy.linalg import svd, norm, inv
from homogropy import toHomoForm, on_mouse, genPointMatrix, DLT, constructH, transForm
WINDOW_NAME = 'window'


def parse_args():
    parser = argparse.ArgumentParser(description="Document Rectification")
    parser.add_argument("Image", type=str, default="images/1-0.png", help="Path to the image to rectify")



# https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
def billinear(image, x, y):
    '''
    This function is to billinearly interpolate pixel value

    Input:
        image: original image to warp
        x: the row value after image transformed
        y: the column value after image transformed

    Reutrn:
        image: interpolated image, same shape as original iamge
    '''
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    wa = np.repeat((y1-y)*(x1-x), 3).reshape((-1, 3))
    wb = np.repeat((x1-x)*(y-y0), 3).reshape((-1, 3))
    wd = np.repeat((x-x0)*(y1-y), 3).reshape((-1, 3))
    wc = np.repeat((x-x0)*(y-y0), 3).reshape((-1, 3))

    return wa*image[x0, y0]+wb*image[x0, y1]+wc*image[x1, y1]+wd*image[x1, y0]




# https://towardsdatascience.com/image-geometric-transformation-in-numpy-and-opencv-936f5cd1d315
def inWarp(origin_img, H):
    '''
    This function is to warp source image 

    Input:
        origin_img: numpy array of original image
        H: Homogrophy to warp the image
    '''
    coord = np.indices((origin_img.shape[0], origin_img.shape[1])).reshape(2, -1).T

    new_img = np.zeros((origin_img.shape[0], origin_img.shape[1], 3)).astype(np.uint8)
    origin_img_ponts_trans = transForm(coord, inv(H))

    xcoord = origin_img_ponts_trans[:, 0]
    ycoord = origin_img_ponts_trans[:, 1]
    indices = np.where((xcoord >= 0) & (xcoord < origin_img.shape[0]) & (ycoord >= 0) & (ycoord < origin_img.shape[1]))
    xpix, ypix = xcoord[indices], ycoord[indices]

    xpixnew, ypixnew = coord[indices, 0], coord[indices, 1]

    new_img[xpixnew, ypixnew] = billinear(origin_img, xpix, ypix)

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.imshow(WINDOW_NAME, new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == '__main__':

    # Parse the argument
    args = parse_args()

    # Read the file
    img = cv.imread(args.Image) 

    # Click the corner of document manually
    points_add = []
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])

    # Visualize the pointed document
    while True:
        img_ = img.copy()
        for i, p in enumerate(points_add):
            # draw points on img_
            cv.circle(img_, tuple(p), 2, (0, 255, 0), -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exits when pressing ESC

    cv.destroyAllWindows()

    print('{} Points added'.format(len(points_add)))

    points1 = np.array(points_add)
    # transverse the coordinate cause cv and numpy problem
    points1[:, [0, 1]] = points1[:, [1, 0]]
    points2 = np.array([[0, 0],[0, img.shape[1]-1],[img.shape[0]-1, img.shape[1]-1],[img.shape[0]-1, 0]])
    points1_homo, points2_homo = toHomoForm(points1, points2)
    A = genPointMatrix(points1_homo, points2_homo)
    h = DLT(A)
    H = constructH(h)
    inWarp(img, H)
    