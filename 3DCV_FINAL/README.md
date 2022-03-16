# 3DCV FINAL Project Group 14 

### Usage
### Segmentaion
   * To test the segmentation please run
    ```
    Python degree_segmentation.py [PATH TO PCD FILE]
    ```
### Classification
   * We convert the mesh model dataset into point cloud we describe in the report 
   * To test the classification please first download this three file, and put them under project folder.
   * We  only support one class, i.e car, now. 
   ```
   class Car file for training https://drive.google.com/file/d/11_frWx_5o33wSw96iUsfrLPhvQgMz33D/view?usp=sharing
   ```
   ```
   class Car file for validation https://drive.google.com/file/d/1mZFDE-yCV-cUKRR2D4sEGPpgVc0pUXnk/view?usp=sharing
   ```
   ```
   class other file for validation https://drive.google.com/file/d/1z-wd963Dz86LIHHnH8pVjWlzsqyLRltw/view?usp=sharing
   ```
  * To train the one-class SVM please run
   ```
   Python anomaly_onclass_SVM.py
   ```
### Crawler
   * To build the download link list of free 3D model from website, please run
  ```python crawler.py ```
  * Please note that we DO NOT use the data of free 3D model in our experiment finally.
