# Camera Relocalization
This porject practice how to relocate the camera loaction by implementing p3p algorithm on paired sift feature from pointcloud and rgb image.

Camera Relocalization include two features:

* Calculate the rgb image rotaion and translation from camera coordinate to word coordinate
* Create a VR video that simulate a cube attach on the front gate of NTU 

## Installation

### Requirements

* Python >= 3.8
* open3d >= 0.12.0
* pandas >= 1.2.3
* scipy >= 1.6.1
* opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree opencv-python==4.5.1.48
* numpy >= 1.19.5

### Dataset

To run the demo, please run the ./download_data.sh script berfore running the demo

## Usage

### Camera rolocalization

#### Using opencv provided pnp solver

```
python3 2d3dmathcing.py 1
```

#### Using implemented p3p with Ransac solver

```
python3 python3 2d3dmathcing.py 2
```

#### some info about my p3p

Ransac setting  
s: 4 e: 0.5 p: 0.99 d: 10

p3p algorithm is using simple trilatiration to solve the localization

### VR cube video


#### Using opencv provided pnp solver

```
python3 VRvideo.py 1
```

#### Using implemented p3p with Ransac solver

```
python3 VRvideo.py 2
```

This video combine the camera relocaliztion and painting algorithm to generate VRvideo with virtual box

![Alt Text](homework2/ARVideo.gif)

## License
The MIT License (MIT)
