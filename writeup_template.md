# **Finding Lane Lines on the Road** 

## Writeup Template by Felipe Rojas


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./test_images/solidWhiteCurve_raw.png "Road lines Solid White"
[image3]: ./test_images/solidYellowCurve_raw.png "Road lines Solid Yellow"
[image4]: ./test_images/solidWhiteCurve_lanedetection.png "Road Pipeline Solid White"
[image5]: ./test_images/solidYellowCurve_lanedetection.png "Road Pipeline Solid Yellow"
---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of seven steps. For each step, I used an independent function, 
1.- First, I converted the images to grayscale,
2.- then I used Gaussian blur to eliminate most non-linearities. 
3.- After that, I used the Canny function for edge detection in the blur image. 
4.- The fourth step is to define an Area of Interest, in which the program will be searching for lines. In this case, I built a polygon of the same shape of the road. I will build a black mask to use over the canny image so I get a new image with only the road lines on it. 
5.- Next step is to use the Hough algorithm to identify the lines in the new image. This function will return an array with all the X and Y points of the detected lines. It is important to try different parameters to see which ones give us better results. 
6.- With the Hough lines, we can calculate and draw lines in a blank image
7.- Finally, we add the color image and the blank image with the drawn lines.

![alt text][image2]
![alt text][image3]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by calculating the slope and Y intercept for each line found by the Hough algorithm.
I classified each line by their slope. Positive slopes are for the right lane and negative slopes are for the left lane. To draw a consistent line, I added the slopes and Y intercepts for each line for all the lines in the Hough array and divided by the total number of lines on each lane. There were cases when the algorithm detected no lines, and
a division by zero error popped up. To solve this I stored the previous slope value and in case no lines were detected I used the previous slope to build the line. Using the
equation for a straight line Y = mX + b, and Y point reference as bottom of the image (540) and Y at the end of the road (337) I calculated the X values. 

![alt text][image4]
![alt text][image5]

#### Challenge 
I encounteres 3 main troubles in the challenge. The first one is that the size of the challenge video is different than the other videos. Since I used raw values for the vertices of the polygon, I changed them to percentages. For example, the left bottom vertice will be 0.05(5%) of the total length of X and 1(100%) of the total lenght of Y.
The second problem was that the car hood was visible in the challenge video, so it was a problem when trying to detect the lines of the road. So, I used less size of Y for the
vertices to omit the car hood. The problem is that I lose some information, and in some cases the algorithm do not detect lines at all in the right lane.
Sometimes horizontal lines where detected, so the X point values tend towards Infinity. I set the limit to draw X values only when these values
are between 0 and the max width of the image.



### 2. Identify potential shortcomings with your current pipeline


The biggest shortcoming I have is that random lines in the road can mess up with the pipeline. Since I calculate the mean of the slopes of all the lines, random lines can
substantially change the mean of the pipelines and thus, tilt the pipeline in wrong directions.


### 3. Suggest possible improvements to your pipeline

To improve my pipeline, I could implement an error reduction algorithm, where if the mean of the slopes of the lines in certain frame of the video has high variance compared
to other means, ignore or correct that error so the pipeline does not tilt in a wrong way.
Another way could be implementing color detection for the program to detect only white and yellow lines on the road. With both edge detection and color detection, we could discard false positives lines on the road and only take in account the lines delimiting 
