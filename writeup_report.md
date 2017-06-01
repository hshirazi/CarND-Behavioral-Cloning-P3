### Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_lane_driving.jpg "Center Lane Driving Image"
[image2]: ./examples/bridge_driving.jpg "Bridge Driving Image"
[image3]: ./examples/normal.jpg "Normal Image"
[image4]: ./examples/flipped.jpg "Flipped Image"

### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* model_run.mp4 the video of the car riving autonomously using the trained moidel.

Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

My model consists of three convolution neural network with 5x5 filter sizes and depths 24, 36, and 48 (model.py lines 47-49), followed by two convolution neural network with 3x3 filter sizes and depths 64 (model.py lines 50-51), followed by three fully connected layers of size 100, 50, and 10, which then mapps to the final output layer of size 1 which is a number insicating the proposed steering angle.

The model includes RELU layers to introduce nonlinearity after each convolution layer (model.py lines 47-51), and the data is normalized in the model using a Keras lambda layer (code line 45). 

### Attempts to reduce overfitting in the model

The model contains enough training data to ensure it is not overfitting. Besides training data from going round the loop, two pass through the bridge was also added to the training data. Since bridge has a different texture than the rest of the road it helped the neural network to learn more about the bridge as well.

The model was trained and validated on the data set (model.py line 58-59). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 58).

### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. In addition to driving all the way through the track, I also used two passes of driving just on the bridge, to make the network more familiar with bridges. I also used flipped images to remove the left turn bias in the training data.

For details about how I created the training data, see the next section. 

### Solution Design Approach

The overall strategy for deriving a model architecture was to different well known architectures, find the one that works best and then tune it to work better for this problem.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it was successful in image recognition using convolutional layers. It did not perform that well. Then I tried the NVIDIA model which performed significantly better but still had issues. I explain below how I fixed them.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the training set and on the validation set. I implied that the model was working well. 

The final step was to run the simulator to see how well the car was driving around track one. The vehicle drove very well up until the bridge, but then it went off track on the bridge. I realized the bridge had very different characteristics than the rest of the road. I concluded that since the length of the bridge was short the amount of training data for the car driving on the bridge was not enough. 

To improve the driving behavior on the bridge I added two passes of the car driving on the bridge to the training data to make the neural network more familiar with the bridge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Final Model Architecture

The final model architecture (model.py lines 44-56) consisted of a convolution neural network with the following layers and layer sizes:

1. Normalization
2. Cropping 70 pixels from top and 25 pixels from bottom to remove non-relevant data.
3. Convolutional layer with filter size of 5x5 and depth of 24 followed by a RELU layer for non-linearity.
4. Convolutional layer with filter size of 5x5 and depth of 36 followed by a RELU layer for non-linearity.
5. Convolutional layer with filter size of 5x5 and depth of 48 followed by a RELU layer for non-linearity.
6. Convolutional layer with filter size of 3x3 and depth of 64 followed by a RELU layer for non-linearity.
7. Convolutional layer with filter size of 3x3 and depth of 64 followed by a RELU layer for non-linearity.
8. Fully connected layer of size 100 applied to the flattened output of previous layer. 
9. Fully connected layer of size 50.
10. Fully connected layer of size 10.
11. Fully connected layer to the output of size 1.

The output specifies the steering angle. So this is a regression problem instead of a classification one.

### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

After realizing the vehicle does not perfom well on the bridge using the trained model, I added two passes of the car just on the bridge to the training data:

![alt text][image2]

To augment the data set even more, I also flipped images and angles thinking that this would remove the left turn bias on the training data, because the track just turns left all the way. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]

After the collection process, I had 16662 number of data points. I then preprocessed this data by normalizing the rgb values to be between -0.5 and 0.5. I also cropped 75 pixles from the top and 25 pixels from the bottom of the image to remove the sky and the car front from the image.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as the model had a very low mean squared error after two epochs and not much change in later epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
