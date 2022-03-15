import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_lineSet(rotation, position, cameraMatrix, epoch, size):

    # get four corner points and camera 
    points = np.array([[0, 0, 1], [1080, 0, 1], [1080, 1920, 1], [0, 1920, 1]])
    points = np.linalg.pinv(cameraMatrix)@points.T
    points =  position.reshape(3, 1) + np.linalg.pinv(rotation)@points
    points = points.T 

    # get center and concate all
    all_points = np.ones((points.shape[0]+1, points.shape[1]))
    all_points[:-1, :] = points

    all_points[-1, :] = position


    # set up line
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(all_points),
        lines = o3d.utility.Vector2iVector([[0,1], [1,2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
    )

    # generate color
    color = [0, 0, 0]
    colors = np.tile(color, (8,1))

    line_set.colors = o3d.utility.Vector3dVector(colors)

    return (line_set, position)

def get_lineSet_opencv(rotation, position, cameraMatrix, epoch, size):

    # get four corner points and camera
    points = np.array([[0, 0, 1], [1080, 0, 1], [1080, 1920, 1], [0, 1920, 1]])
    points = np.linalg.pinv(cameraMatrix)@points.T
    rotation = R.from_rotvec(rotation.reshape(1,3)).as_matrix()[0]
    position = (-np.linalg.pinv(rotation)@position.reshape(3,1)).T
    points =  position.T + np.linalg.pinv(rotation)@points
    points = points.T 

    # get center and concate all
    all_points = np.ones((points.shape[0]+1, points.shape[1]))
    all_points[:-1, :] = points

    all_points[-1, :] = position[0]

    # set up line
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(all_points),
        lines = o3d.utility.Vector2iVector([[0,1], [1,2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
    )

    # generate color
    color = [0, 0, 0]
    colors = np.tile(color, (8,1))

    line_set.colors = o3d.utility.Vector3dVector(colors)

    return (line_set, position[0])



def get_lineTrajectory(cameraOne, cameraTwo):

    # set up line
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector([cameraOne, cameraTwo]),
        lines = o3d.utility.Vector2iVector([[0,1]])
    )

    # generate color
    color = [1, 0, 0]

    line_set.colors = o3d.utility.Vector3dVector([color])

    return line_set

def find_next_pos(current_pos, all_pos):
    next_pos_index = min(range(len(all_pos)), key= lambda theIndex : np.linalg.norm(all_pos[theIndex]-current_pos))
    distance = np.linalg.norm(all_pos[next_pos_index]-current_pos)
    return next_pos_index, distance

def sortThePos(camera_pos):
    sorted_pos = []
    # left_most_pos_index = min(camera_pos, key= lambda theArray : theArray[0])
    left_most_pos_index = max(range(len(camera_pos)), key= lambda theIndex : np.linalg.norm(camera_pos[theIndex]))
    sorted_pos.append(camera_pos[left_most_pos_index])
    del camera_pos[left_most_pos_index]
    # left_pos = [ pos for pos in camera_pos if pos[0] < middle_pos[0]]

    # right_pos = [ pos for pos in camera_pos if pos[0] >= middle_pos[0]]
    
    # left_pos_sort = sorted(left_pos, key= lambda theArray : theArray[2], reverse=True)    

    # right_pos_sort = sorted(right_pos, key= lambda theArray : theArray[0])   

    all_run_time = len(camera_pos)-1

    for i in range(all_run_time):
        left_most_pos_index, distance = find_next_pos(sorted_pos[-1], camera_pos)
        if(distance > 0.8):
            continue
        sorted_pos.append(camera_pos[left_most_pos_index])
        del camera_pos[left_most_pos_index]

    return sorted_pos


def displayPointcloud(points3D, camera, isOpencv):

    # draw point cloud
    points, pointColor = points3D
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pointColor.astype(np.float)/255)

    displaysList = [pcd]

    # draw camera
    all_rotation, all_position = camera
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    

    all_cameraPos = []
    for i in range(len(all_rotation)):
        if(isOpencv):
            oneCamera, cameraPos = get_lineSet_opencv(all_rotation[i], all_position[i], cameraMatrix, i, len(all_rotation))
        else:
            oneCamera, cameraPos = get_lineSet(all_rotation[i], all_position[i], cameraMatrix, i, len(all_rotation))
        displaysList.append(oneCamera)
        all_cameraPos.append(cameraPos)

    sorted_pos = sortThePos(all_cameraPos)

    for i in range(len(sorted_pos)-1):
        traject = get_lineTrajectory(sorted_pos[i], sorted_pos[i+1])
        displaysList.append(traject)

    # traject = get_lineTrajectory(left_sort[len(left_sort)-1], right_sort[0])
    # displaysList.append(traject)

    # for i in range(len(right_sort)-1):
    #     traject = get_lineTrajectory(right_sort[i], right_sort[i+1])
    #     displaysList.append(traject)


    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    displaysList.append(mesh_frame)

    o3d.visualization.draw_geometries(displaysList)

