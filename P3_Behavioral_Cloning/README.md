# **Behavioral Cloning**

## Behavioral Cloning Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/arch.jpg "Model Visualization"
[image2]: ./examples/center.jpg "Center"
[image3]: ./examples/recover1.jpg "Recovery Image"
[image4]: ./examples/recover2.jpg "Recovery Image"
[image5]: ./examples/recover3.jpg "Recovery Image"
[image6]: ./examples/orig.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode (the speed has been modified to 20)
* `model.h5` containing a trained convolution neural network
* `README.md` summarizing the results (this file)

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolutional layers and 3 fully connected layers. The first 3 conv layers use 3x3 filters, (2,2) strides and depths between 24 and 64 (model.py lines 35-40).

The fully connected layers are having sizes 100, 50 and 10 respectively.

The model includes RELU layers to introduce nonlinearity (code line 35-40), and the data is normalized in the model using a Keras lambda layer (code line 31).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 195). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 50).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center lane driving, recovering from the left and right sides of the road, and smooth turning.

For details about how I created the training data, see the next section.

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to explore different models.

My first step was to use a convolution neural network model similar to the one that I used for the traffic sign classification project. I used that model as a baseline. Then I discovered there is the Nvidia's self driving model. So I implemented it and the result look promising.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add dropoff in the fully connected layers, but it doesn't seem to work very well. So my strategy was to monitor the validation errors, when trainnig error decrease but validation error increases, it's a sign of stopping the training. In general, I found that 3 epochs are enough to get reasonable result.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track: it drove off the road when not at the center region of the road, and it also tend to go off to the dirt at the sharp turn immdiately after passing the bridge. To improve the driving behavior in these cases, I collected more data to teach it how to recover from the side of the road as well as more turning data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 24-50) consisted of a convolution neural network with the following layers and layer sizes: an input layer followed by normalization layer and cropping layer. After that, the model passes preprocessed data to the convolutional layers.

There are 5 convolutional layers. All of them uses *ReLU* as the activation function. Filter size of the first 3 layers is *(5, 5)* and last 2 is *(3, 3)*. The stride of the first 3 is *(2, 2)* and the rest is *(1,1)*. The depth of the 5 filters are 24, 36, 48, 64 and 64 respectively.

Fully connected layers are of size 100, 50 and 10. The final output layer is just a float representing the steering angle.

Here is a visualization of the architecture.

![alt text][image1]

_Note: The image is generated using the blew code._

```python
from keras.utils.visualize_util import plot
plot(model, to_file='arch.png', show_shapes=True)
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to getting back to the center when the directions are off. These images show what a recovery looks like starting from the vehicle at the right side of the road and slowly bearing to the center of the road:

![alt text][image3]

![alt text][image4]

![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would make the training data more balanced. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 16,894 data points. I then preprocessed this data by normalizing the color values into range [0,1]. The images are also croped such that the top 70 pixels (above the road) and the bottom 25 pixels (part of the car) are removed.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the fact that after 3rd epoch, there is no significant improvement of the validation accuracy and sometimes it also overfits the training data. I used an adam optimizer so that manually training the learning rate wasn't necessary.
