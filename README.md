# **Project 4 - Advanced Lane Lines**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
---

[//]: # (Image References)

[image1]: ./writeup_images/1_chessboard_corners.png "Chessboard Corners Drawn"
[image2]: ./writeup_images/2_distortion_correction_chessboard.png "Undistorted Chessboard"
[image3]: ./writeup_images/3_distortion_correction.png "Undistorted Image"
[image4]: ./writeup_images/4_color_channels.png "Separate Color Channels"
[image4a]: ./writeup_images/4a_r_binary.png "R-Channel Binary"
[image4b]: ./writeup_images/4b_s_binary.png "S-Channel Binary"
[image4c]: ./writeup_images/4c_sobelx_binary.png "Sobel-X Binary"
[image5]: ./writeup_images/5_combined_binary.png "Combined Thresholded Binary"
[image6]: ./writeup_images/6_warped.png "Perspective Transform"
[image7]: ./writeup_images/7_warped_binary.png "Warped Binary"
[image8]: ./writeup_images/8_masked_binary.png "Masked Warped Binary"
[image9]: ./writeup_images/9_histogram.png "Histogram"
[image10]: ./writeup_images/10_sliding_widows.png "Sliding Window and Polynomial Fit"
[image11]: ./writeup_images/11_lane_pixel_prev_fit.png "Lane Pixel using Previous Fit"
[image12]: ./writeup_images/12_lane_boundary.png "Lane Boundary"
[image13]: ./writeup_images/13_lane_boundary_visual_data.png "Lane Boundary with Visual Data"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---	

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is the **Step 1** of the IPython notebook `P4_AdvancedLaneLines.ipynb`.

I started by preparing "object points", which are the (x, y, z) coordinates of the chessboard corners in the world. Assuming the chessboard is fixed on the (x, y) plane and z=0, the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners are detected in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Below is the array of calibration images with all chessboard corners detected and drawn using `cv2.drawChessboardCorners()` function:

![alt text][image1]

Note: Chessboard corners are not drawn on on Calibration Images 1, 4, and 5 as `cv2.findChessboardCorners` was not able to detect all the interior corners.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. To visualize distortion correction I applied the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

In **Step 2** of the IPython notebook `P4_AdvancedLaneLines.ipynb`, I applied the distortion correction function `cv2.undistort()` to one of the test images like this one:

![alt text][image3]

The effect of distortion correction is subtle, but can be perceived from the difference in shape of the hood of the car at the bottom corners of the image and surrounding tress and white car on the right side of the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Color transform and gradient threshold selection is shown in the **Step 3** of the IPython notebook `P4_AdvancedLaneLines.ipynb`. I first examined separate color channels of RGB and HLS color spaces. 

![alt text][image4]

From the above images of separate channels, we see that R-channel from the RGB color space and S-channel from the HLS color space very well identifies the yellow colored lane line in case of varying gradients of lighting and different road surfaces.

Thus, thresholding the R-channel and S-channel such that both the yellow and white lane lines are identified clearly as shown below.

![alt text][image4a]

![alt text][image4b]

But both R-channel and S-channel fail to detect the lane lines under shadows. Thus, applying Sobel thresholds in x - direction helps to identify the lane lines under shadows as shown below.

![alt text][image4c]

I used a combination of R-channel, S-channel, and Sobel thresholds in x - direction to generate a combined binary image as shown below:

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform includes a function called `warp()`, which is in the **Step 4** of the IPython notebook `P4_AdvancedLaneLines.ipynb`.  The `warp()` function uses `cv2.getPerspectiveTransform()` to transform the image based on the  source (`src`) and destination (`dst`) points. Assuming the road is always relatively flat and the camera position is fixed, a fixed perspective transform may be appropriate. So, I hardcoded the source and destination points in the following manner:

```python
src_bottom_left  = [  205, 720 ]
src_bottom_right = [ 1110, 720 ]
src_top_left     = [  580, 460 ]
src_top_right    = [  705, 460 ]

dst_bottom_left  = [  320, 720 ]
dst_bottom_right = [  960, 720 ]
dst_top_left     = [  320,   0 ]
dst_top_right    = [  960,   0 ]

src = np.float32([src_bottom_left,src_bottom_right,src_top_right,src_top_left])
dst = np.float32([dst_bottom_left,dst_bottom_right,dst_top_right,dst_top_left])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

Also, applied the `warp()` function for perspective transform on binary image to get the warped binary image. 

![alt text][image7]

I also applied region of interest mask to eliminate any pixels other than in the expected region of lanes as shown below.

![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the **Step 5. a)** of the IPython notebook `P4_AdvancedLaneLines.ipynb`, the function `slidingwindow_polyfit()` identifies lane lines and fit a second order polynomial to both right and left lane lines.

First, it computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. These locations are identified from the local maxima of the left and right lanes in the expected region of the histogram based on perspective transform instead of broadly looking in the left of right halves.

```python
leftx_base = np.argmax(histogram[200:500]) + 200
rightx_base = np.argmax(histogram[800:1100]) + 800
```

![alt text][image9]

The function then identifies nine windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image. Pixels belonging to each lane line are identified and the `Numpy.polyfit()` method fits a second order polynomial to each set of pixels. The image below demonstrates how this process works:

![alt text][image10]

In the **Step 5. b)** of the IPython notebook `P4_AdvancedLaneLines.ipynb`, the function `polyfit_using_prev_fit` function performs basically the same task, but alleviates much difficulty of the search process by leveraging a previous fit (from a previous video frame, for example) and only searching for lane pixels within a certain range of that fit. The image below demonstrates this - the green shaded area is the range from the previous fit, and the yellow lines and red and blue pixels are from the current image:

![alt text][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lane and the position of the vehicle with respect to center is calculated in the **Step 6** of the IPython notebook `P4_AdvancedLaneLines.ipynb`. The radius of curvature is based on Chapter 35 of Lesson 15 of Udacity Self Driving Course. The equation below is used to find radius of curvature: 
```python
curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```
where, fit[0] is the first coefficient (the y-squared coefficient) of the second order polynomial fit, and fit[1] is the second (y) coefficient. y_0 is the y position within the image upon which the curvature calculation is based (the bottom-most y - the position of the car in the image - was chosen). y_meters_per_pixel is the factor used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters.

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:
```python
lane_center = (right_fitx[719] + left_fitx[719])/2
center_offset = (car_position - lane_center) * x_meters_per_pix
```
where, right_fitx[719] and left_fitx[719] are the x-intercepts of the right and left fits, respectively, at the maximum y value of 719. Assuming that the camera is mounted at the center of the vehicle, the vehicle center offset is the difference between these intercept points and the image midpoint.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
**Step 7** of the IPython notebook `P4_AdvancedLaneLines.ipynb` performs the inverse transform (unwarp) using the inverse perspective matrix `Minv` and plot back the detected lane boundary back onto the original image using `inverse_transform()` function.

![alt text][image12]

In **Step 8** of the IPython notebook `P4_AdvancedLaneLines.ipynb`, function `visual_data()`
writes text identifying the curvature radius and vehicle position data onto the original image:

![alt text][image13]


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/viralzaveri12/CarND-P4-AdvancedLaneLines/tree/master/output_videos "Project Videos with Lane Boundary and Visual Data")

Folder contains 3 output test videos:
1. project_video_output.mp4
2. challenge_video_output.mp4
3. harder_challenge_video_output.mp4

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, I considered creating a combined binary image by applying gradients and thresholds for only R-channel of RGB color space and and S-channel of HLS color space. Processing the project video always threw error of non-empty vector expected. I realized that both R and S channels are not capable of finding lanes under shadows. Thus, I incorporated Sobel threshold in x - direction to create the combined binary image.

I also applied a region of interest over the warped binary image to mask only the lane lines and convert the surrounding pixels to zero. Without applying region of interest mask, the surrounding detected pixels threw off the lane line detection and radius of curvature and center offset calculations.

The pipeline works well on the project_video.mp4. Lane boundary is detected throughout the video with sensible / realistic radius of curvature values. However, pipeline fails on challenge_video.mp4 and harder_challenge_video.mp4 as lane lines detected are way off. A much better approach is required for lane detection.

If I were to pursue this project in future, I would begin by taking screen shots from the challenge videos and use those as the test images. One thing I noticed is harder_challenge_video.mp4 contains very curvy roads, and the coordinates I used for perspective transform of the lane lines to view "birds-eye-view" would not fit these highly curvy roads. So I would definitely select smaller area of lane lines for perspective transform. Also, I would try gradients and thresholds of different color channels to robustly identify lane lines under different conditions such as bright light, shadows, etc.