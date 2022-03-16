# Visual Odometry
This project practice how to estimate the visual odometry from steam of the video
The project include feature match and estimate camera pose using Epipolar Geometry

Visual Odometry include one feature:

* Generate odometry video by extracting feature from stream of images

## Installation

### Requirements
* Python >= 3.8
* open3d >= 0.12.0
* opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree opencv-python==4.5.1.48
* numpy >= 1.19.5

### Dataset

To run the demo, please run the ./download_data.sh script before running the demo

## Usage

We first need to calibrate our camera, please run

```
python3 camera_calibration.py [CALIBRATE_VIDEO]
```

Note that out demo calibration_video is calib_video.avi

### Visual Odometry

```
python3 vo.py frames
```

## Licencse
The MIT License (MIT)
