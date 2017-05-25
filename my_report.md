## Vehicle Detection Project Report

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report/car_notcar.png
[image2]: ./report/hog_features.png
[image3]: ./report/sliding_window.png
[image4]: ./report/sliding_window_multi.png
[image5]: ./report/consecutive_frame.png
[image6]: ./report/labels_and_bbox.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it! I tried to put my code in different sections using the markdown format in ipython notebook. So it's easy to refer to the position of code from the report.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in Section 1 "Helper Functions" in my ipython notebook. The code of extracting HOG features is in function `get_hog_features`. This function basically calls the `hog` function from `skimage.feature`.

I started by reading in all the `vehicle` and `non-vehicle` images in `Sec 2.0` of the notebook.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that color space has the biggest impact, and the `cells_per_block` the least (in my case). I finally settled to 'YUV' since later I got very good prediction accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

My code for training the model is in Section 2.2 of the ipython notebook. First I need to extract features of the cars and non-cars images. Then I scaled the features using `StandardScaler`. Next I split the data set into training and testing. Finally, I trained a linear SVM using `svc.fit(X_train, y_train)`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In section 1 of the notebook, there is a helper function `slide_window` that generates a set of windows with fixed size and given location. I have to call this function multiple times to generate windows of different size and locations later in my pipeline.

The `search_windows` helper function searches for cars in a give set of windows.

An example image, with all windows that predicts a car in it is given here:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 1-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In `Section 5` of my ipython notebook, I created a class BoxBuffer. It uses a `deque` data structure to store the bounding boxes generated by the most recent `n_frame` frames. For each frame of image we are processing currently, after we use the function `search_windows` to get the `hot_windows`, we call `BoxBuffer.add_boxes(hot_windows)` to add the current hot box in the deque (the oldest hot_window will be popped out automatically). When calling `add_heat` to generate the heatmap, instead of using the current hot_windows as the second parameter (as in the single image case), we use `BoxBuffer.get_buffered_boxes()` to get all the `hot_windows` from all the recent 6 frames.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames and the resulting bounding box on the last frame in the series:
![alt text][image6]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One problem I had was my pipeline was very slow. It processes about 2 frames per second. I have tried my best to keep the feature size small and not to use too dense sliding windows.

I think it'll be nice to use the convolutional neuro network to detect car and compare with the SVN. I'll consider implementing this in the future.