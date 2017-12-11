# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 images
* The size of the validation set is 4,410 images
* The size of test set is 12,630 images
* The shape of a traffic sign image is 32x32 pixels
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

There are 43 labels available in the dataset. Below is a table of each label, as well as its count in each dataset. Note this is directly pulled from the notebook.
```
Speed limit (20km/h) - Train: 180    Test: 60    Valid: 30
Speed limit (30km/h) - Train: 1980    Test: 720    Valid: 240
Speed limit (50km/h) - Train: 2010    Test: 750    Valid: 240
Speed limit (60km/h) - Train: 1260    Test: 450    Valid: 150
Speed limit (70km/h) - Train: 1770    Test: 660    Valid: 210
Speed limit (80km/h) - Train: 1650    Test: 630    Valid: 210
End of speed limit (80km/h) - Train: 360    Test: 150    Valid: 60
Speed limit (100km/h) - Train: 1290    Test: 450    Valid: 150
Speed limit (120km/h) - Train: 1260    Test: 450    Valid: 150
No passing - Train: 1320    Test: 480    Valid: 150
No passing for vehicles over 3.5 metric tons - Train: 1800    Test: 660    Valid: 210
Right-of-way at the next intersection - Train: 1170    Test: 420    Valid: 150
Priority road - Train: 1890    Test: 690    Valid: 210
Yield - Train: 1920    Test: 720    Valid: 240
Stop - Train: 690    Test: 270    Valid: 90
No vehicles - Train: 540    Test: 210    Valid: 90
Vehicles over 3.5 metric tons prohibited - Train: 360    Test: 150    Valid: 60
No entry - Train: 990    Test: 360    Valid: 120
General caution - Train: 1080    Test: 390    Valid: 120
Dangerous curve to the left - Train: 180    Test: 60    Valid: 30
Dangerous curve to the right - Train: 300    Test: 90    Valid: 60
Double curve - Train: 270    Test: 90    Valid: 60
Bumpy road - Train: 330    Test: 120    Valid: 60
Slippery road - Train: 450    Test: 150    Valid: 60
Road narrows on the right - Train: 240    Test: 90    Valid: 30
Road work - Train: 1350    Test: 480    Valid: 150
Traffic signals - Train: 540    Test: 180    Valid: 60
Pedestrians - Train: 210    Test: 60    Valid: 30
Children crossing - Train: 480    Test: 150    Valid: 60
Bicycles crossing - Train: 240    Test: 90    Valid: 30
Beware of ice/snow - Train: 390    Test: 150    Valid: 60
Wild animals crossing - Train: 690    Test: 270    Valid: 90
End of all speed and passing limits - Train: 210    Test: 60    Valid: 30
Turn right ahead - Train: 599    Test: 210    Valid: 90
Turn left ahead - Train: 360    Test: 120    Valid: 60
Ahead only - Train: 1080    Test: 390    Valid: 120
Go straight or right - Train: 330    Test: 120    Valid: 60
Go straight or left - Train: 180    Test: 60    Valid: 30
Keep right - Train: 1860    Test: 690    Valid: 210
Keep left - Train: 270    Test: 90    Valid: 30
Roundabout mandatory - Train: 300    Test: 90    Valid: 60
End of no passing - Train: 210    Test: 60    Valid: 30
End of no passing by vehicles over 3.5 metric tons - Train: 210    Test: 90    Valid: 30
```



### Design and Test a Model Architecture

#### PreProcessing

Before - and after! - considering and building the architecture for my convolutional neural net, preprocessing the images became necessary. For later evaluation images, I included image resizing via opencv to get it to match the dataset's 32x32 image size. The datasets themselves all came in at that size, so this step was unnecessary for them. 

The next step for pre-processing was to normalize the data. This was done to bring pixel value rangs to between -0.5 and 0.5, using 0 to 255 as the range of possible pixel values.
```
def normalize(img):
    min = -0.5
    max = 0.5
    min_pixel = 0
    max_pixel = 255
    return min + ((img - min_pixel) * (max - min)) / (max_pixel - min_pixel)
```

Reshaping the images became neccessary due to the convolutional network expecting a 3 channel image. I used the numpy reshape command to correct the shape for processing use. This originally caused issues until I fixed it.

I also have grayscale functionality built into the notebook, though its usage is commented out in the image preprocessing section. I originally ran the network accepting a one channel grayscale image, but the results would peak in the mid to high 80s on the test set and low 80s on the validation set. This makes sense in retrospect. Color is key to identifying the meaning behind various traffic signs, so removing that data would provide a less robust solution for identifying traffic signs.

After I added the color images back (and reworked the architecture a bit to accept the three channel images) my results became much better.

#### Architecture

I used the LeNet architecture, with convolutional neural networks at the front, but instead of subsampling layers I introduced dropout layers to prevent overfittign.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB normalized image   							| 
| Convolution     	| filter shape: (5,5,3,6) stride: (1,1,1,1), valid padding, outputs 28x28x6, linear function 	|
| RELU					|												|
| Max pooling	      	| kernel: (1,2,2,1) stride: (1,2,2,1),  valid padding outputs 5x5x16 				|
| Dropout         |     Dropout using placeholder value, default 1.0, used 0.3 during training |
| Convolution	    | filter shape: (5,5,6,16) stride: (1,1,1,1) valid padding, outputs 10x10x16, linear function |
| RELU          | |
| Max pooling       |  kernel: (1,2,2,1)   stride: (1,22,1) valid padding output: 5x5x16 |
| Flatten         |     Input: 5x5x16 Output: 400 |
| Fully connected		| Input 400 Output 120  linear function       									|
| Relu				|         									|
| Dropout         |   Dropout using placeholder value, default 1.0, used 0.3 during training |
|	Fully connected			|		Input 120 Output = 84						|
| Relu        |     |
| Dropout         |   Dropout using placeholder value, default 1.0, used 0.3 during training |
|	Fully Connected					|		Input 84 Output = 43										|
 
Results/outputs were then one-hot encoded.

#### Training

To train the model, I used a the Adam Optimizer, aiming to minimize the loss. The Loss function was a soft max crss entropy averaged across training batches.

I trained for 50 EPOCHS with a batch size of 128, a learning rate of 0.001, and a dropout keep rate of 0.3 (so 70% of neurons would be dropped during training). I experimented with several epoch counts, learning rates, and batch sizes. I found that, despite having a GPU that could take huge swaths of the dataset in a singular batch, I found better results with the smaller 128 batch size. 50 Epochs seemed to move me towards an acceptable accuracy percentage (10,20, and 35, my previous attempts, seemed to stop shy of the desired percentage).

#### Problems and solutions

The architecture saved within the notebook and described above was not the original architecture.

The LeNet architecture was originally chosen because it was the suggested architecture for the project. The convolutional layers would work excellently identifying features that may not be positionally important, but important to identify a sign. For instance - detecting curved versus angled sides for a triangular versus circular sign. A feature detector that would identify a yellow sign versus a red, etc.

As mentioned before in the preprocessing section, I originally attempted to train with grayscale images, never getting a satisfactory accuracy percentage. After realizing the importance of color, I switched over to accepting the 3 channel color input.

I added dropout later to deal with overfitting - without the dropout, I would have a 95%+ accuracy on the training set, but low 80% or even 70%ish accuracy on the validation set - a sure sign of overfitting the training data. I initially had success with a default keep rate of 0.5 (50% of neuron passthrough disabled) but when still faced with a lower validation accuracy to training I decreased keep rate to 0.3 (70% of neuron passthrough disabled) until I had a good validation accuracy.

My final model results were:
* training set accuracy of 98.4%
* validation set accuracy of 93.8%
* test set accuracy of 91.7%

### Test a Model on New Images

#### Testing with real world images

I grabbed several signs from google images in order to perform a quick real-world evaluation of german traffic signs. I googled the term of several random labels and grabbed the best image I could find that met the following criteria:

1. Was a picture of a real sign in the real world.
2. Only had that sign in the picture (if this was on a self driving car, location and isolation of the sign would be neccessary prior to identification)
3. Did not have a ton of watermarks (easier said than done)

Once found, I loaded these into the notebook. I should note that I performed the additional task of using OpenCV to resize the images. I also split the images and reordered them back to RGB format, as opencv will convert loaded images to BGR (and the architecture expects RGB).

The output of the sign evaluation was:
```
Results: [33 29 12 14 17]
Expected: [33, 25, 12, 14, 17]
Accuracy: 0.8
```

It had predicted all but one sign accurately. The sign it had miss was road work ahead, which it had incorrectly identified as bicycles crossing. After looking at the two signs, this made sense. In the next section, you can see that the model was unsure on this one, and had three possibilities it thought could be plausible, including the right answer. All three of these signs have the same color and shape. When I resized the images to the 32x32 image, it seemd the road workers became more pixelated, and more blobbish. That blob seemed to make the network think it was likely to be biycles, children crossing, or construction workers working on the road.


#### Softmax probabilities

Here is the softmax probabilities for the real-world image evaluations. This provides both what the model identified the sign to be, and its top 5 probabilities.

```
==========================
Is a Turn right ahead
Was identified as a Turn right ahead
==========================
Highest ranking 5:
0 - Turn right ahead at probability 0.5816525816917419
1 - Go straight or left at probability 0.1346615105867386
2 - Roundabout mandatory at probability 0.13096866011619568
3 - Keep left at probability 0.06874947994947433
4 - Turn left ahead at probability 0.04236813634634018
==========================

==========================
Is a Road work
Was identified as a Bicycles crossing
==========================
Highest ranking 5:
0 - Bicycles crossing at probability 0.9236929416656494
1 - Bumpy road at probability 0.028669096529483795
2 - Road work at probability 0.024066148325800896
3 - Traffic signals at probability 0.012362510897219181
4 - Beware of ice/snow at probability 0.007108269724994898
==========================

==========================
Is a Priority road
Was identified as a Priority road
==========================
Highest ranking 5:
0 - Priority road at probability 1.0
1 - No passing for vehicles over 3.5 metric tons at probability 6.366930020738269e-20
2 - Ahead only at probability 9.154691362044581e-25
3 - Traffic signals at probability 7.532705672579112e-27
4 - End of no passing by vehicles over 3.5 metric tons at probability 7.5329889768739315e-28
==========================

==========================
Is a Stop
Was identified as a Stop
==========================
Highest ranking 5:
0 - Stop at probability 1.0
1 - No vehicles at probability 2.983049191777193e-13
2 - Yield at probability 2.090915865132054e-15
3 - No passing for vehicles over 3.5 metric tons at probability 1.4228148428067417e-17
4 - No entry at probability 1.7190392009846126e-18
==========================

==========================
Is a No entry
Was identified as a No entry
==========================
Highest ranking 5:
0 - No entry at probability 0.9999744892120361
1 - Stop at probability 2.548661723267287e-05
2 - Traffic signals at probability 4.7735446373703763e-14
3 - No passing for vehicles over 3.5 metric tons at probability 1.6543552297631757e-23
4 - Priority road at probability 4.1840712486016036e-27
==========================

```
