
# Using Computer Vision and Machine Learning on a Raspberry PI based GoPiGo car to follow a line.

[//]: ![GPG](./media/cover_photo.JPG)

I have given presentations on machine learning, where one of the interactive examples that I use is to train a simulated car driving on a simulated road.  

I thought, could I use what I know about machine learning and a GoPiGo to build an actual device that followed a line.

I realize DexterIndustries has a ['line follower' sensor](https://www.dexterindustries.com/GoPiGo/gopigo-line-follower-getting-started/using-gopigo-line-follower/) but the goal was to use computer vision techniques to acquire and process the raw image data, into a form that a machine learning model can learn from.

While this project is simple in concept, I found it be a very good intermediate level, end-to-end, machine learning project.

The article represents my approach to solving that problem. 

## Audience

Who is the audience for this article? 

My target audience for this article are educators that are looking for a fun way to introduce the concepts of computer vision and machine learning to students and those people that like to work with a RaspberryPI from a hobbyist perspective.

This is not a hard hitting oeuvre on the latest advances in computer vision and machine learning.  But if you are looking for a fun and accessible application of the technology then I think you will enjoy this article.

## Overview

This project uses a GoPiGo3 Robot Car from DexterIndustries with a RaspberryPI3 - not even the latest RaspberryPI4 - with an additional Raspberry PI camera, and a GrovePI button.

The goal was to create a training track with different kinds of turns and line curvatures to collect training images that represent:

* Left Turns

* Right Turns

* Continue Straight 

This is a supervised machine learning project, meaning we will collected labeled training data.

Once we have enough labeled training data, we can start the process of finding the 'best' machine learning model.  In this case, 'best' is not just the one that is the most accurate predictor but also the one that can run fast enough for the GoPiGo to make decisions.  

Lastly, once we have a trained model, create a new testing track to see how the model behaves on a track it has not seen before.

Once the GoPiGo was fully deployed, I also wanted a way to stream the video that the car was 'seeing' back to a laptop to get a sense for how the model was behaving.

### Considerations

* Speed of training

We will see that the ultimate model had to be trained on the RaspberryPI.  While some models and techniques might have produced better accuracy, the trade off of training time had to be considered.

* Speed of making predictions

While the car was driving, the video camera was feeding frames to the computer vision pipeline and model to turn predictions.  This process had to be as fast as possible.  The speed of model inference was a very important characteristic to consider.

* Streaming the video from the GoPiGo to a laptop dashboard

To provide insight into what the GoPiGo was seeing, and the decisions it was making, I wanted to stream the video back to a laptop to be displayed.  This turned out to be very helpful in the 'debugging' phase but also visually engaging after the system was functional.


## Materials
### GoPiGo Robot Car
[GoPiGo](https://www.dexterindustries.com/gopigo3/)

### Floor Mat

I used 9, 1/2 inch black form floor tiles that you can find inexpensively at [Walmart](https://www.walmart.com/ip/Everyday-Essentials-1-2-Thick-Flooring-Puzzle-Exercise-Mat-with-High-Quality-EVA-Foam-Interlocking-Tiles-6-Piece-24-Sq-Ft-Multiple-Colors/336366651?selected=true)

You want to use black, because the contrast between the floor and the line will be very important.

### Line Tape

I used basic masking tape and found it to work well but white electrical tape would also work.  Keep in mind that you will be putting the tape down and taking it up often and you want that to be simple and non-destructive to the floor mats.

### Raspberry PI Camera

You will want the V2 Camera Module.  The [RaspberryPI](https://www.raspberrypi.org/products/camera-module-v2/) org has a page with a number of places to get this camera.

### Grove Button ( Optional )

[Grove Button](https://shop.dexterindustries.com/shop/sensors-accessories/sensors-actuators/grove-button)

I used a button to start and stop the GoPiGo car but this is not required.

### Camera Mount

The camera mount is tricky because, as far as I know, there is no commercial camera mount for the GoPiGo3 that attaches to the front of the car at about a 45 degree angle pointing downwards.

I used a camera mount like:

[Amazon Link to Camera Mount](https://www.amazon.com/Makeronics-Acrylic-Holder-Raspberry-Transparent/dp/B07SQL2RNR)

but that tilts the camera up.  The rounded points of the base of mount just fit into a couple of the GoPiGo3 holes.  I then tipped the camera mount forward, put some weather stripping under it, and used a twist tie to fasten it to the GoPiGo3.  I know - its very McGyver.

The angle is about 45 degress from the horizontal.  

Adrian in the RPi4CV Hacker Bundle used some duct tape and cardboard.  

> Left Side
![Camera Left](./media/camera_mount/camera_left.JPG)

> Right Side
![Camera Left](./media/camera_mount/camera_right.JPG)

> Top
![Camera Left](./media/camera_mount/camera_top.JPG)

> Bottom
![Camera Left](./media/camera_mount/camera_bottom.png)


## Car Setup

## RaspberryPI Image
Getting a properly configured RaspberryPI image can be challenging.

I started with the Desktop Buster version of [Raspbian](https://www.raspberrypi.org/downloads/raspbian/).

First let me say I **DO NOT** recommend the GoPiGo DexterOS.  It is too limiting and for this project you will want full control to install software.

You will need to include OpenCV on the RaspberryPI. Here are some links and suggestions to get started:

### My Blog Post

I created a [Medium](https://medium.com/@patrick_ryan/building-opencv-4-10-on-raspian-buster-and-raspberry-pi4-64669bd2eb74) blog post on installing OpenCV4 on a RaspberryPI.  It is time consuming but if you follow the steps it is not hard.

### PyImageSearch.com

[PyImageSearch](https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/) has a blog post on how to install OpenCV4 on a RPI4, but it should work the same with a RPI3.

### PyImageSearch RaspberryPI for Computer Vision

[RaspberryPI for Computer Vision](https://www.pyimagesearch.com/raspberry-pi-for-computer-vision/)

This is a 3 book set, and you can buy 1,2 or all 3 books.  With any book purchase Adrian provides a Raspian Image with the libraries already setup and installed.  This is by far the easiest way to get started, and you get an amazing set of books on how to use the Rasbperry PI for computer vision.

In the Hacker bundle, he has a GoPiGo line following example where he uses 'fuzzy logic' to follow the line.


## Collecting Training Data

### Image as zeros and ones

## Training a model

## Deploying a model to the RaspberryPI

### Adding the button

### Starting the program at startup

## Testing the Line Follower

## Things I learned

### Start/Stop turning versus continuous turning

### Train model on the Raspberry PI

### Speed of car and turning speed

### Review your training data

## References

### ImageZMQ

### PyImageSearch

### Dexter Industries


## ZMQ

`pip install pyzmq`

## Run Image Server

`source ~/.virtualenvs/py36cv4_venv/bin/activate`

`python receive_images.py`

`python server_training_data_collector.py --save-images 1`


## RPI

`source ~/.virtualenvs/gopigo3/bin/activate`

`python ./send_immediate_images.py --server-ip 192.168.1.208`

* You have to train the model on the RPI because you cannot train and save the model on one architecture and load it from another cpu architecture.

* You have to transfer the training images to rpi

* train model and save model

## Train Model

Training the model is handled by a different project

/Users/patrickryan/Development/python/mygithub/gpg3-linefollow-model

This will save a model file that can be loaded