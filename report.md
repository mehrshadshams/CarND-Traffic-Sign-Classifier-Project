# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[train_sample]: ./images/train_sample.png "Training Sample"
[valid_sample]: ./images/valid_sample.png "Validation Sample"
[test_sample]: ./images/test_sample.png "Test Sample"
[statistics]: ./images/train_data_per_class.png "Data set  statistics"
[augmented]: ./images/augmented_sample.png "Augmented Data"
[augmented_single]: ./images/augmentation_single_sample.png "Augmented and it's original"
[history_5]: ./images/history5.png "Training History"
[history]: ./images/history_final.png "Training History (final)"
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[f1_recall]: ./images/f1_recall.png "F1/Recall"
[image4]: ./custom/1.png "Traffic Sign 1 (Double Curve)"
[image5]: ./custom/2.png "Traffic Sign 2 (Keep Left)"
[image6]: ./custom/3.png "Traffic Sign 3 (No entry)"
[image7]: ./custom/4.png "Traffic Sign 4 (Children Crossing)"
[image8]: ./custom/5.png "Traffic Sign 5 (End of all speed and passing limits)"
[vis_weight1]: ./images/visualize_weights1.png "Visualization of weights 1st CNN layer"
[vis_weight2]: ./images/visualize_weights2.png "Visualization of weights 2nd CNN layer"
[vis_weight3]: ./images/visualize_weights3.png "Visualization of weights 3rd CNN layer"
## Rubric Points

### DataSet exploration
The first thing I did was to visualize the images by randomly select a few samples from each of the training/validation/test data sets. Also I visualized the distribution of images across different classes

### Preprocessing
The first thing I did was to compute the mean of all images in the training set, across each color channel, and subtracted that value from the validation and test set. Then I converted the data to float32 and normalized it between zero and one. The first version of the network architecture, which was basically the LeNet-5, I trained it on color images. 

Later I decided to tackle the architecture explained in the baseline paper by Pierre Sermanet and Yann LeCun and for that for each image in the training set I applied the 6 following transformations randomly and added the result image to the training set. 

* Rotation [-15..15] degrees
* Horizontal translation [-0.05..0.05] of the width of the images
* Vertical translation [-0.05..0.05] of the height of the images
* Scaling [0.9..1.1]
* Shear [0.1]
* Random intensity change [0.6..0.8] or original intensity

Without the data augmentation it was hard to get higher accuracy. Having augmented data also helps with the generalization of the model. To create fake data I used a little handy utility in Keras library called [ImageDataGenerator](https://keras.io/preprocessing/image/) which generates random transformations.

I also used pandas library to calculate summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

### Visualization

![alt text][statistics]

Here you can see a sample of the augmented data 

![alt text][augmented]

![alt text][augmented_single]

### Model Architecture

I used 3 models for this project. First I used plain old vanila LeNet-5 architecture to get a baseline accuracy

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|		
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| Output size: 5x5x16=400 						|
| Fully connected		| Output size: 400x120 						|
| Fully connected		| Output size: 84x43 						|
| Softmax				|         									|

I trained the network a couple of times for 100 epochs and got training accuracy around 98% but the model was over-fitting and validation accuracy was about %89.2-%91.4

Next I increased the model capacity and added Dropout to mitigate over-fitting. I used this architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x100 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x200 	|		
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x200 				|
| Dropout	      	| keep probability: 0.5 				|
| Flatten		| Output size: 5x5x200=5000 						|
| Fully connected		| Output size: 5000x1000 						|
| Fully connected		| Output size: 1000x200 						|
| Fully connected		| Output size: 200x43 						|
| Softmax				|         									|

With this architecture and training for 80 epochs I was able to get up to 99% training accuracy and %95.2 validation accuracy:

![alt text][history_5]

Next I tried to implement the architecture described in the paper. The paper describes that they got better performance using only gray scale images, so that's what I did.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| (1) Dropout           | keep probability: 0.9 	                    |		
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 16x16x64 	|		
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				    |
| (2) Dropout           | keep probability: 0.8 	                    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 8x8x128 	|		
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 				    |
| (3) Dropout           | keep probability: 0.7 	                    |
| (a) Flatten		    | Output size: 4x4x128=2048 					|
| (b) Flatten output (1)| Output size: 16x16x32=8192 					|
| (c) Flatten output (2)| Output size: 8x8x64=4096  					|
| Concat a,b,c          | Output size: 14336                            |
| Fully connected		| Output size: 14336x1024 						|
| Fully connected		| Output size: 1024x43   						|
| Softmax				|         									    |

And this is the result I got:

![alt text][history]

### Model training

I used Adam optimizer with the learning decay with batch size of 128 for 50 epochs for my final model.

### Results

As I mentioned above I started with a simple model (LeNet) with base pre-processing to get a baseline for the expected results. Then I increased the model capacity and introduced Dropout as a measure of controlling over-fitting in order to increase accuracy and generalization at the same time. For better performance I also used data augmentation techniques mentioned above.

My final results are:

|               | Accuracy | 
| ------------  | -------- | 
| Training      | %99.9    |
| Validation    | %96.8    | 
| Test          | %95.9    | 

To better understand how model behaves I created the plot of F1 score and Recall

![alt text][f1_recall]

## Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

All of these images have a different pose than the training set images, ie the sign is tilted or has an unusual pose. 

Here are the results of the prediction:

|| Image			        |     Prediction	        					| 
|:--|:---------------------:|:---------------------------------------------:| 
| ![alt text][image4] | Double curve      	| Road narrows on the right   				| 
| ![alt text][image5] | Keep left     		| Keep left 					|
| ![alt text][image6] | No entry			| No entry  					|
| ![alt text][image7] | Children crossing	| Children crossing 			|
| ![alt text][image8] | End of all speed and passing limits		| Speed Limit (30km/h)  |

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares not to favorably to the accuracy on the test set of 96.8%. I think the model needed more epochs to train.

For the first image, the model was unsure between the "Double curve" and "Road narrows on the right" as we see in the tables below.
For the last image, I think the unusual pose of the sign is the reason that the model was not able to predict correctly.

For the first image, the model is relatively sure that this is a road narrows on the right sign (probability of 0.97), but it's a double curve.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Road narrows on the right   									| 
| .19     				| Double Curve 										|
| 0.0004					| Children crossing											|
| 0.00014	      			| Road work					 				|
| .000006				    | Bicycles crossing      							|


For the second image, model is completely sure about the prediction.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep left   									| 
| 7e-20     				| Go straight or left 										|
| 2.9e-20					| Turn right ahead											|
| 3.7e-21	      			| Stop					 				|
| 1.55e-24				    | Roundabout mandatory      							|

For the third image model is completely sure about the prediction. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| No entry   									| 
| 0.0076     				| No passing 										|
| 5.84240552e-05					| Stop											|
| 2.0747395e-05	      			| Keep right					 				|
| 1.35515547e-05				    | Priority road      							|

For the fourth image model is completely sure about the prediction.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Children crossing    									| 
| 2.00412712e-13     				| Dangerous curve to the right										|
| 4.01074742e-14					| Beware of ice/snow 											|
| 2.27944325e-14	      			| Road narrows on the right					 				|
| 8.34595115e-15				    | Right-of-way at the next intersection      							|

For the fifth image, model completely make a mistake.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)    									| 
| 5.24470316e-08    				| Speed limit (70km/h) right										|
| 1.68708447e-09					| Speed limit (20km/h) 											|
| 5.59473891e-11	      			| Speed limit (50km/h) right					 				|
| 3.7612774e-11				    | Keep right      							|

I created the visualization of the layers. The image I used for this part is "Speed limit (60km/h)"

1st convolution layer:
The model learns the contours on the sign with different edges
![alt text][vis_weight1]

2nd convolution layer:
![alt text][vis_weight2]

3rd convolution layer:
![alt text][vis_weight3]
