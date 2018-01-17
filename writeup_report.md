# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sample_images/center_track1_original.jpg "Center Image Track1"
[image2]: ./sample_images/center_track2_original.jpg "Center Image Track2"
[image3]: ./sample_images//center_track1_flipped.jpg "Track 1 flipped Image"
[image4]: ./sample_images//center_track2_flipped.jpg "Track 2 flipped Image"
[image5]: ./sample_images//data_dis.png "Data Distribution Image"
[image6]: ./sample_images//loss.png "Loss Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_trial10_v5_rgb_newdata_secondtrack_included.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Yes you can use the above code to check but change it to following lines in my case!! Yes it took many many trials and versions!!! Phewwww!! 
```
python drive.py model_trial10_v1_rgb_newdata_secondtrack_included.h5
```
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First, I started with the LeNet architecture and understood the working and flow. Then I switched to the Nvidia model. At some point of time, I used dropout with keepalive = 0.5 to avoid overfitting but finally I removed it and didn't use dropout at all. 

I have used Lambda layer to normalize the data with the input shape of 160 x 320 x 3 (Original Image shape)
Then I have used cropping layer.

It is followed by five convolution layers.
The model includes RELU layers to introduce nonlinearity, 


#### 2. Attempts to reduce overfitting in the model

The model does not contains any dropout layers to reduce overfitting

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of two laps Forward/backward driving on track 1 and forward driving on track 2.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First, I started with the basic LeNet architecture and only center images to get an overall idea about the behaviour. I realized that I have to get more data, perform some augmentation technique and use more powerful architecture than LeNet. Udacity has suggested one excellent Nvidia paper to follow. So I decided to go with Nvidia architecture for this project. 

I wanted to move in step by step mode and improve my model little bit every time. So I followed these steps:

1.	LeNet architecture with Center images and Udacity data
2.	LeNet architecture with Center images and my first set of data
3.	LeNet architecture with Center images and Udacity + my first set of data
4.	LeNet architecture with Center/left/right images and both data
5.	Nvidia architecture with Center/left/right images and both data
6.	Nvidia architecture with Center/left/right images with my second set of data

I have tried many versions for above six steps which includes:
	-	YUV,RGB,BGR color space
	-	Balancing data by removing data points which are near to zero
	-	Augmentation techniques like Flipping, brightness adjustments, resizing images
	-	RELU vs ELU
	-	Dropout/ Without dropout

I worked in two parallel ideas:
1.	Only Udacity data with more augmentation techniques
2.	Track1 + Track2 data with less augmentation techniques

After trying many number of times and tuning parameters, second idea worked for me.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

My model was performing well without dropout so I didn't use it for my second idea.

The final step was to run the simulator to see how well the car was driving around track one. I returne the color space and correction angle to improve this model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes

y complete architecture in tabular form is as follow:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3					  					| 
| Lambda		     	| x/255.0 - 0.5								 	|
| Cropping				| cropping=((50,20), (0,0))						|
| convolution 		    | 24 - 5x5 										|
| RELU                  |                                               |
| convolution 	    	| 36 - 5x5									 	|
| RELU					|												|
| Convolution		    | 48 - 5x5										|
| RELU                  |                                               |
| Convolution 	    	| 64 - 5x5										|
| RELU					|												|
| Convolution 		    | 64 - 5x5										|
| RELU                  |                                               |
| Flatten               | 							                    |
| Fully connected		| 100						    			    |
| Fully connected	 	| 50		         							|
| Fully connected		| 10					    			        |
| Fully connected		| 1        										|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using forward direction and two laps in reverse direction. Here is an example image of center lane driving:

![alt text][image1]

I then recorded one lap on track two in forward direction. Here is an example image of track 2 image.

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would would balance my data. As we have more left turn in track 1 than right turn. For example, here is an image that has then been flipped:

Track 1 fliiped image example:

![alt text][image3]

Track 2 fliiped image example:

![alt text][image4]

After the collection process, I had around 22K number of data points.
My data distribution looks like following:

![alt text][image5]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.

My training and validation data loss look like following:

![alt text][image6]

Conclusion: I learned a lot from this project. I feel that the most important thing in this project is DATA. QUALITY DATA!!! I thought that the more data will help me but I was wrong. I should have applied solid augmented technique on udacity data or should have tried to collect good data earlier. I think I panicked at the middle stage of my project and started making my model more and more complex with more data! It was not necessary at ALL! I made it little bit harder! May be from reading too many articles, forum answers etc. BUT, the ultimate truth is I LEARNED A LOT! 
