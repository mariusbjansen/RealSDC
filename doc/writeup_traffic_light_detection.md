# Traffic light detection
One task in the final project of the Udacity Nanodegree Self-Driving Car Engineer was to stop the automated car at stop lines corresponding to traffic lights.

This led to the subtasks

* find the next stop line ahead
* detect the current state of a traffic light

## Find the next stop line ahead
For the first subtask there was already a nice tutorial called “Detection Walkthrough” on the Udacity website by Eren and Steven. The code is written in the file tl_detector.py and mainly resides in the function process_traffic_lights. The function compares the current position of the car with the planned waypoints and then checks for the next stop line corresponding to a traffic light. So eventually the interesting outcome is the distance to the next stop line.

## Detect the current state of a traffic light
After that the traffic light state needs to be determined. The outcome of the classification (labels) are the following.

* red
* yellow
* green
* unknown (optionally)

The input to this task is video data. So eventually images from the video stream need to be classified. We thought of different approaches.

<img src="traffic_light_detection_architectures.png" width=640>

We looked into options b and c and finally chose tbd.

### Traffic light detection pipeline

#### Tensorflow Object Detection API 

[GitHub Repository Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
The tesnorflow object detection API allows to localize and identify multiple objects in a single image. It is very well maintained and happily developed and used by Google employees. It has already been introduced in projects for the Udacity nanodegree by several teams successfully. So we also decided to have a look into it.

The API can be used with different models from the so called object detection model zoo. [GitHub Repository Tensorflow object detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). These models in the zoo are basically detection models pre-trained on datasets like the COCO dataset. [Website COCO dataset](http://cocodataset.org).

Happily the COCO dataset already contains traffic lights and the models are pretrained on them.

There is the possibility to train locally (from scratch or transfer learning) [Link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md). Therefore images need to be in the tfrecord format. This is documented [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md).

In order to create our own dataset based on the simulator and Udacity data we already provided [this script](https://github.com/mcounter/RealSDC/blob/master/Test_Images/Simulator/test/conv2tfrec.py) in order to convert png or jpg images with label data XML files in PASCAL VOC format for exampled labelled with the tool [labelImg](https://github.com/tzutalin/labelImg)

We tried out different models from the zoo

* [which is ssd_mobilenet_v1_coco_2017_11_17](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) which was the default from the object detection jupyter notebook 
* [ssdlite_mobilenet_v2_coco_2018_05_09](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz) and 
* [ssd_mobilenet_v2_coco_2018_03_29](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29)

We found sdlite_mobilenet_v2_coco_2018_05_09 the best trade off between accuracy and speed.


#### How HSV works
#### Layout of CNN color detection of traffic lights
#### Reason we chose a specific solution

