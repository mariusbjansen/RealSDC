# Traffic light detection
One task in the final project of the Udacity Nanodegree Self-Driving Car Engineer was to stop the automated car at stop lines corresponding to traffic lights.

This leads to the subtasks

* find the next stop line ahead
* detect the current state of a traffic light

## Find the next stop line ahead
For the first subtask there was already a nice tutorial called “Detection Walkthrough” on the Udacity website by Eren and Steven. The code is written in the file tl_detector.py and mainly resides in the function process_traffic_lights. The function compares the current position of the car with the planned waypoints and then checks for the next stop line from a traffic light. So eventually the interesting outcome is the distance to the next stop line.

## Detect the current state of a traffic light
After that the traffic light state needs to be determined. It is all based on video data. So an image needs to be classified. We thought of different approaches.

<img src="traffic_light_detection_architectures.png" width=640>

We looked into options b and c and finally chose tbd.

### TODO
#### How the google object detection API works very briefly
#### How HSV works
#### Layout of CNN color detection of traffic lights
#### Reason we chose a specific solution

