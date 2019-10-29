# **Self driving car ROS controller** 

## Report


[//]: # (Image References)

[rosArchitecture]: ./reportImgs/rosModel.PNG "System Architecture Diagram"
[fullReal]: ./reportImgs/fullReal.png "Full image from real run"
[cropReal]: ./reportImgs/cropReal.png "Cropped image from real run"
[resizeReal]: ./reportImgs/resizeReal.png "Resized image from real run"
[fullSim]: ./reportImgs/fullSim.png "Full image from simulation run"
[cropSim]: ./reportImgs/cropSim.png "Cropped image from simulation run"
[resizeSim]: ./reportImgs/resizeSim.png "Resized image from simulation run"
---
### Reflection

#### 1. Goal
The aim of this project is to implement waypoint updater node, controller package (DBW Node), traffic detection node [project code](https://github.com/jakubkid/ROS_Full_Self_Driving_Car/tree/master/ros/src)

![ROS Architecture][rosArchitecture]

### Waypoint updater

#### 1. Goal

This node will publish 200 waypoints from the car's current position. It should also update waypoint speed to stop at the stop line when traffic light is red or yellow. It subscribes to '/current_pose' '/base_waypoints' '/current_velocity', '/traffic_waypoint' topics and publishes to 'final_waypoints' topic.

#### 2. Implementation

[Waypoint updater code](https://github.com/jakubkid/ROS_Full_Self_Driving_Car/blob/master/ros/src/waypoint_updater/waypoint_updater.py) implements get_closest_waypoint_idx which finds closest waypoint to current vehicle position next generate_lane checks if vehicle should stop in next 200 waypoints which is done in decelerate_waypoints
decelerate_waypoints first estimates if vehicle can stop in time and continues with speed limit if it is no possible. When it is possible it will aim to decelerate with 0.8 of maximum breaking force 

### Controller package (DBW Node)

#### 1. Goal

This node will calculate brake, throttle and steering based desired speed and position to '/vehicle/steering_cmd', '/vehicle/throttle_cmd'  and '/vehicle/brake_cmd' based on desired speed published in '/twist_cmd' and current speed '/current_velocity', PID algorithm is restarted when controller is restarted when controller is disabled '/vehicle/dbw_enabled'. 

#### 2. Implementation

[Controller package](https://github.com/jakubkid/ROS_Full_Self_Driving_Car/tree/master/ros/src/twist_controller) dbw_node.py is a base file which subscribes to topics and and publishers. All control is done in self.controller.control implemented in twist_controller.py
throttle is controlled with PID controller with kp = 0.3 ki = 0.1 kd = 0.05 to make reaction fast with minimal overshoot. steering angle is controlled by yaw_controller.py and brake is just calculated in Nm from desired speed with respect to maximum brake.


### Traffic detection

#### 1. Goal

This node detects red and yellow light.

#### 2. Data set summary

Training data was annotated manually by me and committed to [repository](https://github.com/jakubkid/ROS_Full_Self_Driving_Car/tree/master/ros/src/tl_detector/light_classification). Real data was extracted from [bag file](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) simulated data was extracted from manually driving in simulator environment.
Because of that training and validation set is quite small
* Real number of training images: 423
* Real number of valid images: 140
* Simulation number of training images: 286
* Simulation number of valid images:  94

#### 3. Real image preprocessing:

Before real image is inputted to the [model](https://github.com/jakubkid/ROS_Full_Self_Driving_Car/blob/master/ros/src/tl_detector/light_classification/tl_classifier_real.py) it is preprocessed to decrease its size. It is first cropped to 30:270 and 0:572 and then it is shrinked to  (60, 144)

|Full image                                                                             |
|![Full image from real run][fullReal]                                                  |
|:-------------------------------------------------------------------------------------:|
|Cropped image                             |  Resized image                             |
|:----------------------------------------:|:------------------------------------------:|
|![Cropped image from real run][cropReal]  |  ![Resized image from real run][resizeReal]|

#### 4. Real Model Architecture.

Real model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 60x144x3 RGB image   					        | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 56x140x6 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride, outputs 28x70x6.					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs20x62x16  	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride, outputs 10x31x16					|
| Flatten Layer1&2		| 10x31x16 = 4960					     		|
| Fully connected		| 4960->120    									|
| RELU					|												|
| Fully connected		| 120->84    									|
| sigmoid				|												|
| Fully connected		| 84->43    									|
 


#### 5. Model training

To train the model, I used AdamOptimizer to minimize loss operation of training set. I set batch size to 1 and with 20 epochs with learning rate with 0.001:

>EPOCH 1 ...
>Validation Accuracy = 0.686
>
>EPOCH 2 ...
>Validation Accuracy = 0.714
>
>EPOCH 3 ...
>Validation Accuracy = 0.693
>
>EPOCH 4 ...
>Validation Accuracy = 0.864
>
>EPOCH 5 ...
>Validation Accuracy = 0.879
>
>EPOCH 6 ...
>Validation Accuracy = 0.829
>
>EPOCH 7 ...
>Validation Accuracy = 0.864
>
>EPOCH 8 ...
>Validation Accuracy = 0.886
>
>EPOCH 9 ...
>Validation Accuracy = 0.893
>
>EPOCH 10 ...
>Validation Accuracy = 0.879
>
>EPOCH 11 ...
>Validation Accuracy = 0.879
>
>EPOCH 12 ...
>Validation Accuracy = 0.800
>
>EPOCH 13 ...
>Validation Accuracy = 0.886
>
>EPOCH 14 ...
>Validation Accuracy = 0.907
>
>EPOCH 15 ...
>Validation Accuracy = 0.871
>
>EPOCH 16 ...
>Validation Accuracy = 0.900
>
>EPOCH 17 ...
>Validation Accuracy = 0.907
>
>EPOCH 18 ...
>Validation Accuracy = 0.893
>
>EPOCH 19 ...
>Validation Accuracy = 0.907
>
>EPOCH 20 ...
>Validation Accuracy = 0.879
>

After training model is stored as [real.ckpt](https://github.com/jakubkid/ROS_Full_Self_Driving_Car/blob/master/ros/src/tl_detector/light_classification/trainedModel/)

#### 6. Simulation image preprocessing:

Before simulation image is inputted to the [model](https://github.com/jakubkid/ROS_Full_Self_Driving_Car/blob/master/ros/src/tl_detector/light_classification/tl_classifier_sim.py) it is preprocessed to decrease its size. It is first cropped to 0:550 and 100:500 and then it is shrinked to  (100, 72)

|Full image                                                                                      |
|![Full image from simulation run][fullSim]                                                      |
|:----------------------------------------------------------------------------------------------:|
|Cropped image                                  |  Resized image                                 |
|:---------------------------------------------:|:----------------------------------------------:|
|![Cropped image from simulation run][cropSim]  |  ![Resized image from simulation run][resizeSim]|

#### 7. Simulation Model Architecture.


Simulation model consisted of the following layers:

| Layer					|		Description								|
|:---------------------:|:---------------------------------------------:|
| Input					| 100x72x3 RGB image   							|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 96x68x6 	|
| RELU					|												|
| Max pooling 2x2		| 2x2 stride, outputs 48x34x6.					|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 40x26x16	|
| RELU					|												|
| Max pooling 2x2		| 2x2 stride, outputs 20x13x16					|
| Flatten Layer1&2		| 20x13x16 = 4160								|
| Fully connected		| 4160->120										|
| RELU					|												|
| Fully connected		| 120->84										|
| sigmoid				|												|
| Fully connected		| 84->43										|
 


#### 8. Model training

To train the model, I used AdamOptimizer to minimize loss operation of training set. I set batch size to 1 and with 20 epochs with learning rate with 0.001:

>EPOCH 1 ...
>Validation Accuracy = 0.713
>
>EPOCH 2 ...
>Validation Accuracy = 0.723
>
>EPOCH 3 ...
>Validation Accuracy = 0.819
>
>EPOCH 4 ...
>Validation Accuracy = 0.862
>
>EPOCH 5 ...
>Validation Accuracy = 0.862
>
>EPOCH 6 ...
>Validation Accuracy = 0.851
>
>EPOCH 7 ...
>Validation Accuracy = 0.872
>
>EPOCH 8 ...
>Validation Accuracy = 0.894
>
>EPOCH 9 ...
>Validation Accuracy = 0.926
>
>EPOCH 10 ...
>Validation Accuracy = 0.904
>
>EPOCH 11 ...
>Validation Accuracy = 0.915
>
>EPOCH 12 ...
>Validation Accuracy = 0.926
>
>EPOCH 13 ...
>Validation Accuracy = 0.926
>
>EPOCH 14 ...
>Validation Accuracy = 0.926
>
>EPOCH 15 ...
>Validation Accuracy = 0.936
>
>EPOCH 16 ...
>Validation Accuracy = 0.947
>
>EPOCH 17 ...
>Validation Accuracy = 0.947
>
>EPOCH 18 ...
>Validation Accuracy = 0.947
>
>EPOCH 19 ...
>Validation Accuracy = 0.947
>
>EPOCH 20 ...
>Validation Accuracy = 0.947
>

After training model is stored as [simulation.ckpt](https://github.com/jakubkid/ROS_Full_Self_Driving_Car/blob/master/ros/src/tl_detector/light_classification/trainedModel/)

