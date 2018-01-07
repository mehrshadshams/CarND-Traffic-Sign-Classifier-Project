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
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

### Model training

I used Adam optimizer with the learning decay with batch size of 128 for 80-100 epochs

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


