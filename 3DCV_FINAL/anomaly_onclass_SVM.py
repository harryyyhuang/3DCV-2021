# import sklearn
import cv2
import numpy as np
import os
import open3d as o3d
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import joblib


pcd_path = r'./pcd2'
validation_path = [r'./other', r'./car']

use_density = False
save_model = False
load_model = True


def pcd_processing(points, folder='none'):
    if folder == 'valid':
        temp = points[:, 1].copy()
        points[:, 1] = points[:, 2]
        points[:, 2] = temp
    if folder == 'train':
        # data augmentation
        if np.random.random_sample() < 0.4:
            temp = points[:, 0].copy()
            points[:, 0] = points[:, 1]
            points[:, 1] = temp
    out = points.copy()
    min_ = min(points[:, 0].min(), points[:, 1].min())
    max_ = max(points[:, 0].max(), points[:, 1].max())
    for i in range(3):
        if i == 2:
            out[:, i] = (out[:, i] - points[:, i].min()) / (points[:, i].max() - points[:, i].min())
        else:
            out[:, i] = (out[:, i] - min_) / (max_ - min_)

        out[:, i] = (out[:, i] - 0.5) * 2

    return out


def scale_to_255(a, min=-2, max=2, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def point_cloud_2_birdseye_density(points,
                                   res=0.1,
                                   side_range=(-8., 8.),  # left-most to right-most
                                   fwd_range=(-8., 8.),  # back-most to forward-most
                                   height_range=(-2., 2.),  # bottom-most to upper-most
                                   ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    points = np.array([x_points, y_points, z_points]).T
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max, 2], dtype=np.uint8)

    # intensityMap = np.zeros((y_max, x_max))
    densityMap = np.ones(z_points.shape)

    _, indices, counts = np.unique(points[:, 0:2], axis=0, return_index=True, return_counts=True)
    # PointCloud_top = points[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    # intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top.max(axis=1)
    densityMap[indices] = normalizedCounts

    # intensityMap = scale_to_255(intensityMap)
    densityMap = scale_to_255(densityMap)
    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img, 1] = pixel_values
    # im[:, :, 0] = intensityMap
    im[y_img, x_img, 0] = densityMap
    # im[y_img, x_img, 2] = densityMap

    return im


def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=(-2., 2.),  # left-most to right-most
                           fwd_range=(-2., 2.),  # back-most to forward-most
                           height_range=(-2., 2.),  # bottom-most to upper-most
                           ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im

if __name__ == "__main__":
    pcd_set = []
    label = []
    count = 0
    for root, dir, files in os.walk(pcd_path):
        # read pcd
        for file in files:
            pcd = o3d.io.read_point_cloud(os.path.join(root, file))
            # convert to numpy
            # o3d.visualization.draw_geometries([pcd])
            pcd = np.asarray(pcd.points)
            pcd = pcd_processing(pcd, 'train')
            # pcd = removePoints(pcd)
            if use_density:
                pcd = point_cloud_2_birdseye_density(pcd)
            else:
                pcd = point_cloud_2_birdseye(pcd)
            # pcd = pcd.transpose((1, 2, 0))
            # pca = PCA(n_components=1000)
            # pcd = pca.fit_transform(pcd.T)

            # cv2.imwrite("./train/"+file+".png", pcd.astype(np.uint8))
            pcd = pcd.flatten()
            # put in pcd set
            # exit(0)
            pcd_set.append(pcd)
            label.append(1)
            count += 1
            if count % 100 == 0:
                print(count)
                # break
    # negative example for training
    # for root, dir, files in os.walk(validation_path):
    #     for file in files:
    #         break
    #         count += 1
    #         if count % 27 == 0 and count % 10 != 0:
    #             pcd = o3d.io.read_point_cloud(os.path.join(root, file))
    #             # o3d.visualization.draw_geometries([pcd])
    #             pcd = np.asarray(pcd.points)
    #             pcd = pcd_processing(pcd)
    #
    #             if use_density:
    #                 pcd = point_cloud_2_birdseye_density(pcd)
    #             else:
    #                 pcd = point_cloud_2_birdseye(pcd)
    #
    #             # cv2.imwrite("./valdiate/negative_" + file + ".png", pcd)
    #             pcd = pcd.flatten()
    #
    #             pcd_set.append(pcd)
    #             label.append(-1)
    #

    pcd_set = np.array(pcd_set)
    print("feature selection")
    sel = VarianceThreshold(threshold=2.5)
    print("original: ", pcd_set.shape)

    if load_model:
        sel = joblib.load('feature_selection_model.pkl')
        pcd_set = sel.transform(pcd_set)
    else:
        pcd_set = sel.fit_transform(pcd_set)

    if save_model:
        joblib.dump(sel, 'feature_selection_model.pkl')

    label = np.array(label)

    print("feature selection: ", pcd_set.shape)
    nu = np.count_nonzero(label - 1) / label.shape[0]
    clf = OneClassSVM(gamma='scale', kernel='rbf', tol=1e-3, nu=0.05, verbose=False)

    if load_model:
        clf = joblib.load('oneClassSVM_model.pkl')
    else:
        clf.fit(pcd_set)

    predict = clf.predict(pcd_set)
    if save_model:
        joblib.dump(clf, 'oneClassSVM_model.pkl')

    print("*" * 10 + "training result" + "*" * 10)
    # print(clf.score_samples(pcd_set))
    print(clf.get_params())
    print("accuracy: ", 1 - (np.count_nonzero(predict - label) / len(predict)))
    print("*" * 30)

    print("start validation")

    del pcd_set
    validation_set = []
    label = []
    count = 0

    print("read negative examples...")
    for root, dir, files in os.walk(validation_path[0]):
        for file in files:
            count += 1
            if count % 10 == 0:
                pcd = o3d.io.read_point_cloud(os.path.join(root, file))
                # o3d.visualization.draw_geometries([pcd])
                pcd = np.asarray(pcd.points)
                pcd = pcd_processing(pcd, 'valid')

                if use_density:
                    pcd = point_cloud_2_birdseye_density(pcd)
                else:
                    pcd = point_cloud_2_birdseye(pcd)
                # cv2.imwrite("./valdiate/negative_" + file + ".png", pcd)
                pcd = pcd.flatten()

                validation_set.append(pcd)
                label.append(-1)

    print("read positive examples...")
    # validation_set = np.array(validation_set)
    # validation_set = sel.transform(validation_set)
    # predict = clf.predict(validation_set)
    # label = np.array(label)
    # print("accuracy: ", 1 - (np.count_nonzero(label - predict) / label.shape[0]))

    # validation_set = []
    # label = []

    for root, dir, files in os.walk(validation_path[1]):
        for file in files:
            # break
            pcd = o3d.io.read_point_cloud(os.path.join(root, file))
            # o3d.visualization.draw_geometries([pcd])
            pcd = np.asarray(pcd.points)
            pcd = pcd_processing(pcd, 'valid')
            if use_density:
                pcd = point_cloud_2_birdseye_density(pcd)
            else:
                pcd = point_cloud_2_birdseye(pcd)

            # cv2.imwrite("./valdiate/positive_" + file + ".png", pcd.astype(np.uint8))
            pcd = pcd.flatten()

            validation_set.append(pcd)
            label.append(1)
            count += 1

    print("predict validation set...")
    label = np.array(label)
    validation_set = np.array(validation_set)
    validation_set = sel.transform(validation_set)
    print(validation_set.shape)
    predict = clf.predict(validation_set)

    print("accuracy: ", 1 - (np.count_nonzero(label - predict) / label.shape[0]))
