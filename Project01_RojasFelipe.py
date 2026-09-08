################################################################################    
################## SELF DRIVING CAR ENGINEER NANO DEGREE #######################
################################################################################

##### Project one: Computer vision. Detecting the lanes of a road
##### by Felipe Rojas\
##### In this project, you will use the tools you learned about in the lesson 
##### to identify lane lines on the road.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2 ## OPEN CV library

def gray_scale(img):
    #Transform the Color 3D image into gray scale 2D image
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size = 5):
    #We Use the blur the remove non linearities in the image
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    #We use canny for Edge detection in our image
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interst(img, vertices):
    # Build a mask of desired shape and size for of region of interest
    mask = np.zeros_like(img)
    
    if len(img.shape)>2:
        channel_count = img.shape[2]
        ignore_mask_color = (255, ) * channel_count
        
    else:
        ignore_mask_color = 255
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #plt.imshow(mask, cmap = 'gray')
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, selector, rho, theta, threshold, min_line_len, max_line_gap):
    # We need a Canny image as input
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, 
                            maxLineGap = max_line_gap)
    img_lines = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    "Selector: 0  for detecting broken lines in a road and 1 for drawing the line over the road"
    if(selector == 0):
        draw_lines(img_lines, lines)
    else:
        draw_lanes(img_lines, lines)
        
    return img_lines


def draw_lines(img, lines, color=[255, 0,0], thickness = 10):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color, thickness)
       
            
       
def draw_lanes(img, lines, color=[0, 0,255], thickness = 12):
    
    m = 0
    b = 0
    #variables for the left lane
    m_added_left = 0
    b_added_left = 0
    left_counter = 0
    global last_left_m
    global last_left_b
    #variables for the right lane
    m_added_right = 0
    b_added_right = 0
    right_counter = 0
    global last_right_m
    global last_right_b
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            #Calculate the slope to know if the lane corresponds to left lane or right lane
            #Positive slope corresponds to right lane and negative slope corresponds to left lane
            m = (y2-y1)/(x2-x1)
            b = y2 - (m*x2)
            
            if m<0:
                m_added_left += m
                b_added_left += b
                left_counter += 1
                       
            else:
                m_added_right += m
                b_added_right += b
                right_counter += 1
                
    # Calculate the average slope and points for EACH LANE
    #LEFT LANE
    #There are cases when NO lines are detected so the counter is 0. We have to make sure
    #that we dont divide by 0 and also, we always draw a lane
    
    #Ocassionally, the edge detection will detect lines other than the road lines.
    #By creating global average slope variables the variance of the slope will be
    #almost regular in all iterations
    global average_left_m
    global average_left_b
    global average_right_m
    global average_right_b

    if left_counter != 0:
        average_left_m = (m_added_left/left_counter)
        average_left_b = (b_added_left/left_counter)
        last_left_m = average_left_m
        last_left_b = average_left_b
    else:
        average_left_m = last_left_m
        average_left_b = last_left_b
    
    
    imshape = img.shape

    Y_bottom = int(imshape[0])
    Y_top = int(imshape[0]*0.63)
    X_bottom = (Y_bottom - average_left_b)/average_left_m
    X_top = (Y_top - average_left_b)/average_left_m
    if((X_bottom>= 0) and (X_bottom<= 960) and ((X_top>= 0) and (X_top<= 960))):
        cv2.line(img,(int(X_bottom),Y_bottom),(int(X_top),Y_top),color, thickness)     

    #RIGHT LANE     
    if right_counter != 0:
        average_right_m = m_added_right/right_counter
        average_right_b = b_added_right/right_counter
        last_right_m = average_right_m
        last_right_b = average_right_b
    else:
        average_right_m = last_right_m
        average_right_b = last_right_b
         
    X_bottom = (Y_bottom - average_right_b)/average_right_m
    X_top = (Y_top - average_right_b)/average_right_m
    if((X_bottom>= 0) and (X_bottom<= 960) and ((X_top>= 0) and (X_top<= 960))):
        cv2.line(img,(int(X_bottom),Y_bottom),(int(X_top),Y_top),color, thickness) 
              

def add_weighted(img, color_image, alpha = 0.8, beta = 1):
    #we draw the lines that we found over the original image
    return cv2.addWeighted(color_image, alpha, img, beta, 0)

# Function that follows step by step the lane detection method
def lane_detection(img, selector=1):
    #Create a copy of the original image
    color_image = np.copy(img)
    #First, we transform the color image to a 2D gray scale image
    gray = gray_scale(img)
    plt.imshow(gray, cmap = 'gray')
    #Then, we apply Gaussian bluring to the image
    blur_image = gaussian_blur(gray, 7)
    plt.imshow(blur_image, cmap = 'gray')
    #Now we use the Canny method for Edge detection
    canny_image = canny(blur_image, 50, 150)
    plt.imshow(canny_image, cmap = 'gray')
    
    #We define the vertices for our Area of Interest Polygon
    '''vertices = np.array([[(48, 540),    #bottom left vertice
                          (436, 337),   #top left vertice
                          (533, 337),   #top right vertice
                          (955,540)]])  #bottom right vertice'''
    imshape = img.shape
    vertices = np.array([[(int(imshape[1]*0.05), int(imshape[0]*0.92)),     #bottom left vertice (X, Y)
                          (int(imshape[1]*0.47), int(imshape[0]*0.60)),     #top left vertice 
                          (int(imshape[1]*0.47), int(imshape[0]*0.60)),     #top right vertice
                          (int(imshape[1]*0.99), int(imshape[0]*0.92))]])   #bottom right vertice   
    
    #Ignore everything but the Area of interest in our Canny image
    masked_image = region_of_interst(canny_image, vertices)
    
    
    # Define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 35 #minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    
    #Identify and draw the lines
    lines = hough_lines(masked_image, selector, rho, theta, threshold, min_line_length,  
                        max_line_gap)
    lanes_image = add_weighted(lines, color_image, 0.8, 1)
    plt.imshow(lanes_image)
    return lanes_image
        

################################################################################    
########### Testing the Lane detection function for Broken lines ###############
################################################################################

image1 = mpimg.imread('test_images/solidWhiteCurve.jpg')
plt.imshow(image1)
image1_broken_lines = lane_detection(image1, 0)
mpimg.imsave("test_images/solidWhiteCurve_raw.png", image1_broken_lines)

image2 = mpimg.imread('test_images/solidWhiteRight.jpg')
plt.imshow(image2)
image2_broken_lines = lane_detection(image2, 0)
mpimg.imsave("test_images/solidWhiteRight_raw.png", image2_broken_lines)

image3 = mpimg.imread('test_images/solidYellowCurve.jpg')
plt.imshow(image3)
image3_broken_lines = lane_detection(image3, 0)
mpimg.imsave("test_images/solidYellowCurve_raw.png", image3_broken_lines)

image4 = mpimg.imread('test_images/solidYellowCurve2.jpg')
plt.imshow(image4)
image4_broken_lines = lane_detection(image4, 0)
mpimg.imsave("test_images/solidYellowCurve2_raw.png", image4_broken_lines)


image5 = mpimg.imread('test_images/solidYellowLeft.jpg')
plt.imshow(image5)
image5_broken_lines = lane_detection(image5, 0)
mpimg.imsave("test_images/solidYellowLeft_raw.png", image5_broken_lines)


image6 = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
plt.imshow(image6)
image6_broken_lines = lane_detection(image6, 0)
mpimg.imsave("test_images/whiteCarLaneSwitch_raw.png", image6_broken_lines)


################# Using the Lane detection function for videos #################

from moviepy.editor import VideoFileClip

white_output = 'test_videos_output/solidWhiteRight_raw.mp4'
yellow_output = 'test_videos_output/solidYellowLeft_raw.mp4'

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
clip2 = VideoFileClip("test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(lane_detection) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
yellow_clip = clip2.fl_image(lane_detection) #NOTE: this function expects color images!!
yellow_clip.write_videofile(yellow_output, audio=False)


############################## C H A L L E N G E ##############################
challenge_output = 'test_videos_output/challenge_raw.mp4'
#clip3 = VideoFileClip("test_videos/challenge.mp4").subclip(0,2)
clip3 = VideoFileClip("test_videos/challenge.mp4")
challenge_clip = clip3.fl_image(lane_detection) #NOTE: this function expects color images!!
challenge_clip.write_videofile(challenge_output, audio=False)
    

################################################################################    
###### Testing the Lane detection function to draw pipelines in the lanes ######
################################################################################

img = mpimg.imread('test_images/solidWhiteCurve.jpg')
plt.imshow(img)
image1_broken_lines = lane_detection(img, 1)
mpimg.imsave("test_images/solidWhiteCurve_lanedetection.png", image1_broken_lines)

image2 = mpimg.imread('test_images/solidWhiteRight.jpg')
plt.imshow(image2)
image2_broken_lines = lane_detection(image2, 1)
mpimg.imsave("test_images/solidWhiteRight_lanedetection.png", image2_broken_lines)

image3 = mpimg.imread('test_images/solidYellowCurve.jpg')
plt.imshow(image3)
image3_broken_lines = lane_detection(image3, 1)
mpimg.imsave("test_images/solidYellowCurve_lanedetection.png", image3_broken_lines)

image4 = mpimg.imread('test_images/solidYellowCurve2.jpg')
plt.imshow(image4)
image4_broken_lines = lane_detection(image4, 1)
mpimg.imsave("test_images/solidYellowCurve2_lanedetection.png", image4_broken_lines)


image5 = mpimg.imread('test_images/solidYellowLeft.jpg')
plt.imshow(image5)
image5_broken_lines = lane_detection(image5, 1)
mpimg.imsave("test_images/solidYellowLeft_lanedetection.png", image5_broken_lines)


image6 = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
plt.imshow(image6)
image6_broken_lines = lane_detection(image6, 1)
mpimg.imsave("test_images/whiteCarLaneSwitch_lanedetection.png", image6_broken_lines)


################# Using the Lane detection function for videos #################

from moviepy.editor import VideoFileClip

white_output = 'test_videos_output/solidWhiteRight.mp4'
yellow_output = 'test_videos_output/solidYellowLeft.mp4'

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
clip2 = VideoFileClip("test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(lane_detection) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
yellow_clip = clip2.fl_image(lane_detection) #NOTE: this function expects color images!!
yellow_clip.write_videofile(yellow_output, audio=False)


############################## C H A L L E N G E ##############################
challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip("test_videos/challenge.mp4")
challenge_clip = clip3.fl_image(lane_detection) #NOTE: this function expects color images!!
challenge_clip.write_videofile(challenge_output, audio=False)
