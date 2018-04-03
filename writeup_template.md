----------


----------


# **Behavioral Cloning** 

## Writeup report for project 3, term 1.

---
This is the report for project 3, term 1. To collect training and validation data, a [simulator develop by Udacity team guys was used](https://github.com/udacity/self-driving-car-sim).
In particular, fast Deep Learning Ne


In this project a convolutional deep neural network model was used to predict human driving bahaviour. 

. More specifically, the model was trained to classify images from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

It was used [anaconda](https://www.anaconda.com/) Python flavour (version 3.6.1), [scikit-learn](http://scikit-learn.org) (version 0.18.1), [TensorFlow GPU](https://www.tensorflow.org/) (version 1.3.0) and [OpenCV](https://opencv.org/releases.html) (version 3.4.0).

For the solution proposed, the next techniques were applied:

* [Contrast Limited Adaptive Histogram Equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html).
* [Data normalization](https://arxiv.org/pdf/1705.01809.pdf).
* [Shuffle training set](http://ieeexplore.ieee.org/document/8246726/?reload=true).
* [Batch Training](https://arxiv.org/abs/1711.00489).
* [Cross correlation](https://arxiv.org/abs/1309.5388).
* [Cross entropy](https://icml.cc/Conferences/2005/proceedings/papers/071_CrossEntropy_MannorEtAl.pdf).
* [Backpropagation](http://yann.lecun.com/exdb/publis/pdf/lecun-88.pdf).
* [Stochastic gradient based optimization](https://arxiv.org/abs/1412.6980).

This is a non exclusive list of openCV functions I used:
* [cv2.createCLAHE()](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html). Used for image contrast enhancement by applying adaptive histogram equalization.

* [cv2.resize()](https://docs.opencv.org/3.4.0/da/d6e/tutorial_py_geometric_transformations.html). Used for rescaling images to 32x32x3 size.

This is a non exhaustive list of Tensorflow functions I used:
* [tf.placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder). Used to feed in the input image in tensor representation.
* [tf.global_variables_initializer()](https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer). Used to initialize all trainable variables.
* [tf.ConfigProto()](https://www.tensorflow.org/programmers_guide/using_gpu). Used to set up GPU memory usage upper boundary.
* [tf.train.AdamOptimizer()](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer). Used for stochastic gradient-based optimization.
* [tf.nn.softmax_cross_entropy_with_logits()](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits). Used to compute cross-entropy between *output network scores* and *expected claseification labels*. However this function is deprecated. Notice that I used this function because I am working with *TensorFlow version 1.3*, but, for *TensorFlow 1.6 and above* you may want to use [tf.nn.softmax_cross_entropy_with_logits_v2](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2) instead.
* [tf.nn.conv2d()](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d). Used to compute the cross correlation between kernels and receptive fields.
* [tf.nn.max_pool()](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool). Used for [downsampling](https://web.stanford.edu/class/cs448f/lectures/2.2/Fast%20Filtering.pdf) each incoming feature map and reduce trainable variables at the same time.
* [tf.nn.relu()](https://www.tensorflow.org/api_docs/python/tf/nn/relu). Used as non-linear activation function, especifically a [Rectified Linear Unit](https://arxiv.org/abs/1611.01491), for model solution proposed.
* [tf.Session()](https://www.tensorflow.org/programmers_guide/graphs). Used to run tensor operations on the static computational graph.
* [tf.train.Saver()](https://www.tensorflow.org/programmers_guide/saved_model). Used to save and restore model variables. Which in fact, retrieves values from the checkpoints using C,C++ libraries under the hood.
* [tf.nn.dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout). Used as regularization by dropping out units (both hidden and visible) in the model using pseudo-random probability depending of [tf.set_random_seed](https://www.tensorflow.org/api_docs/python/tf/set_random_seed) to form random seeds.






**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTUyMDYwNjYwMV19
-->