#**Behavioral Cloning** 

##Writeup Template


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./recover1.jpg "Recovery Image"
[image3]: ./recover2.jpg "Recovery Image"
[image4]: ./recover3.jpg "Recovery Image"
[image5]: ./center.jpg "Normal Image"
[image6]: ./center_flip.jpg "Flipped Image"
[image7]: ./mse.png "Mean Squared Error"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1_20mph.mp4 video of the a complete run of track 1 at a set speed of 20 mph

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 41-53) 

The model includes RELU layers to introduce nonlinearity (code line 44), and the data is normalized in the model using a Keras lambda layer (code line 42). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 60). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 59).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, focusing on smooth curves and driving in the opposite direction.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to begin with a very basic model. Thus, my first step was to use a neural network model with only one fully connected layer. This was only to test the code and the interface to the simulator.
It turned out to be problematic, because the output of the network was between -1..1, but the simulator only showed -25° or +25°. After a deep search I found a hint that this is a known issue on systems, where the decimal period was changed to a comma.
So, an English keyboard layout had to be loaded. It finally worked, but the performance of the model was not good. In a next step, I used the LeNet architecture, which was used to classify traffic signs in the last project.
I thought this model might be appropriate because it showed a good performance in recognizing objects and edges in images. However, this task is a regression task.
The model performance was much better, but it failed in the first curve.
In the next step, I followed the lessons, and implemented the Nvidia architecture. The performance was pretty good, the car was centered in the lane, but there were a few spots where the vehicle fell off the track.
To improve the driving behavior in these cases, I trained especially the particular cases by repeating the curves slowly and smoothly. I also trained the model how to react when approaching the right or left side of the lane by recovering to the middle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 41-53) consisted of a "Nvidia" convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 160x320x3 RGB Image |
| Normalization         | Lambda Layer |
| Cropping              | (70,20)|
|Convolution Layer      |(24, 5, 5) with Relu activation |
|Convolution Layer      |(36, 5, 5) with Relu activation |
|Convolution Layer      |(48, 5, 5) with Relu activation |
|Convolution Layer      |(64, 3, 3) with Relu activation |
|Convolution Layer      |(64, 3, 3) with Relu activation |
|Flatten                |
|Fully Connected Layer | Output = 100 |
|Fully Connected Layer | Output = 50 |
|Fully Connected Layer | Output = 10 |
|Fully Connected Layer | Output = 1 |

Here is a visualization of the architecture.

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving, two in the normal direction and two in the opposite direction. Here is an example image of center lane driving:

![alt text][image5]

I recorded then the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn what to do when it approaches the lanes. These images show what a recovery looks like starting from the left side of the lane and recovering to the middle of the lane :

![alt text][image2]

![alt text][image3]

![alt text][image4]


To augment the data sat, I also flipped images and angles thinking that this would generate more usable training data. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]


After the collection process, I had 16772 number of data points. I then preprocessed this data by normalizing the data to a range of -0.5..0.5. Additionally I cropped (70,25) of the image since this area contains the important information about the road and the lane.
The preprocessing is done directly in the model, so that the images don't have to be be preprocessed separately in the autonomous mode any more.



I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the diagram shown below.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

The validation loss is slightly higher than the training loss, which could indicate overfitting. Thus, I tried to include a dropout layer. However, the validation loss turned out to be higher afterwards.
I didn't use a training generator, since my graphic card (GTX 1060 6 Gb) was able to handle the data.

![alt text][image7]