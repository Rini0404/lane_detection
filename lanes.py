import cv2 as cv
import numpy as np

# python script that does the folowing:
# 1. reads in the image
# 2. converts the image to grayscale
# 3. applies a gaussian blur
# 4. applies the canny edge detector
# 5. applies a mask to the image
# 6. applies the hough transform to the image
# 7. displays the image with the detected lane lines

def output_canny(image):
    gray = cv.cvtColor(lane_image, cv.COLOR_RGB2GRAY)

    blur = cv.GaussianBlur(gray, (5,5), 0)

    canny = cv.Canny(blur, 50, 150)

    return canny
  

def display_lines(image, lines):
  line_image = np.zeros_like(image)
  if lines is not None:
    for line in lines:
      print(line)
      x1, y1, x2, y2 = line.reshape(4)
      cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 15)
  return line_image
  

# return the enclosed region of interest, triangle
def region_of_interest(image):
  height = image.shape[0]
  polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
  ])
  #creates an array of zeros with the same shape and type as a given array 
  mask = np.zeros_like(image)
  # take a triangle and fill it with white
  cv.fillPoly(mask, polygons, 255)
  # bitwise and operation
  masked_image = cv.bitwise_and(image, mask) 
  return masked_image


image = cv.imread('test_image.jpg')

lane_image = np.copy(image)

canny = output_canny(image)

cropped_image = region_of_interest(canny)

# 1st param: image, 2nd param: rho, 3rd param: theta, 4th param: threshold, 5th param: placeholder array, 6th param: placeholder array, 7th param: minLineLength, 8th param: maxLineGap
# threshold: minimum number of votes (intersections in a given grid cell)

empty_array = np.array([])

lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, empty_array, minLineLength=40, maxLineGap=5)

line_image = display_lines(lane_image, lines)

# 1st param: image, 2nd param: alpha, 3rd param: beta, 4th param: gamma
# 2nd multiply the line image by 0.8
# 3rd multiply the original image by 1
# 4th add 1
combine_images = cv.addWeighted(lane_image, 0.8, line_image, 1, 1)

cv.imshow('result', combine_images)

cv.waitKey(0)

