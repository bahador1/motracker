[cars-yolo-output]: examples/assets/cars.gif "Sample Output with YOLO"
[cows-tf-ssd-output]: examples/assets/cows.gif "Sample Output with SSD"



# Multi-object trackers in Python
This implementation is about designing a robust __multi-object tracking (MOT) framework__.

`YOLOv3 + CentroidTracker` |  `TF-MobileNetSSD + CentroidTracker`
:-------------------------:|:-------------------------:
![Cars with YOLO][cars-yolo-output]  |  ![Cows with tf-SSD][cows-tf-ssd-output]
Video source: [link](https://flic.kr/p/L6qyxj) | Video source: [link](https://flic.kr/p/26WeEWy)


The general architecture of this work includes three main subsections:
1.	object detections (creating sets of bounding box coordinates for each detected object in each input frame)
2.	Creating a unique ID for each of the initial detections
3.	tracking module, which tracks each of the objects as they move around frames in a video, maintaining the assignment of unique IDs


Furthermore, object tracking allows us to apply a unique ID to each tracked object, making it possible to count individual objects in a video. 
To build a real-time object tracking algorithm, we should consider the algorithm's speed improvement besides maintaining its accuracy. As a result, our object tracking module will:
1.	Only require the object detection at the first frame that it appears (i.e., when the object is initially detected)
2.	Be able to handle when the tracked object "disappears" or moves outside the boundaries of the video frame.
3.	Be robust to occlusion.
4.	Be able to pick up objects it has "lost" in-between frames.
We implement centroid tracking with OpenCV as the fundamental of our object tracking methodology.
It is easy to understand yet highly effective tracking algorithm, one of the core kernel-based and correlation-based tracking algorithms.

## 1. Object Detection

This module provides the feed of the centroid tracking modules.
we detect moving objects in each video frames and drawing a bounding box or region of the interest (ROI) box around them,
The following models were used as object trackers in this study:


*   SSDMobileNet
*   SSDMobilenetV2
*   YOLOv3
*   HOG+SVM

The best module in terms of accuracy was YOLOv3 (Joseph Redmon et al. 2018), as was expected.
The “You Only Look Once,” or Yolov3, as described in its original paper.  family of models is a series of end-to-end deep learning models designed for fast object detection, developed by Joseph Redmon et al. and first described in the 2015 paper titled “You Only Look Once: Unified, Real-Time Object Detection.”
The approach involves a single deep convolutional neural network (originally a version of GoogLeNet, later updated and called DarkNet based on VGG) that splits the input into a grid of cells. Each cell directly predicts a bounding box and object classification. A result is a large number of candidates bounding boxes that are consolidated into a final prediction by a post-processing step.
There are four main variations of the approach; they are YOLOv1, YOLOv2, YOLOv3, and YOLOv4. The first version proposed the general architecture, whereas the second version refined the design and made use of predefined anchor boxes to improve the bounding box proposal. Version three further refined the model architecture and training process.
They are famous for object detection because of their detection speed, often demonstrated in real-time on video or with camera feed input.


## 2. Centroid Tracking
This object tracking algorithm relies on the Euclidean distance between (1) existing object centroids (i.e., objects the centroid tracker has already seen before) and (2) new object centroids between subsequent frames in a video.
We'll review the centroid algorithm in more depth in the following section. From there, we'll implement a Python class to contain our centroid tracking algorithm and then create a Python script to run the object tracker and apply it to input videos.
Finally, we'll run our object tracker and examine the results, noting both the positives and the algorithm's drawbacks.

`The overall architecture` 
:-------------------------:|
![Cars with YOLO][cars-yolo-output]  
*Fig2*: In our object tracking with Python and OpenCV example, we have a new object that wasn't matched with an existing object, so it is registered as object ID #3


## Available Multi Object Trackers

```
CentroidTracker
IOUTracker
CentroidKF_Tracker
SORT
```

## Available OpenCV-based object detectors:

```
detector.TF_SSDMobileNetV2
detector.YOLOv3
```


## Installation

```
git clone https://github.com/bahador1/motracker
cd multi-object-tracker
pip install -r requirements.txt
pip install -e .
```
## Pretrained object detection models

You will have to download the pretrained weights for the neural-network models using shell scripts provided below `pre_trained models` folder. 

  
  
## How to use it

you can run the code both from jupyter notebook provided below `notebooks folder` or through command line running:

```
python <mot_*.py> --help
```
If your don't pass video's URL to the interpreter, it get video input from your webcam. otherwise you should run the code in the following manner. 
```
python mot_YOLOv3.py --video ../video_data/jamming.mp4
```
