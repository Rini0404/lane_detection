import cv2 as cv
import numpy as np

def make_coordinates(image, line_parameters):
  slope, intercept = line_parameters
  # y1 = image.shape[0] # bottom of the image
  # y2 = int(y1*(3/5)) # slightly lower than the middle
  # y1 = int(image.shape[0] * (3/5))
  # y2 = int(image.shape[0] * (3/5))
  y1 = 0
  y2 = image.shape[0]
  x1 = int((y1 - intercept)/slope)
  x2 = int((y2 - intercept)/slope)
  return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
  left_fit = []
  right_fit = []
  for line in lines:
    # reshape the line to a 1D array = [x1, y1, x2, y2] 
    x1, y1, x2, y2 = line.reshape(4)
    # returns a tuple of the slope and y-intercept
    parameters = np.polyfit((x1, x2), (y1, y2), 1)
    slope = parameters[0]
    intercept = parameters[1]
    if slope < 0:
      left_fit.append((slope, intercept))
    else:
      right_fit.append((slope, intercept))
  # average the values of the left and right lines
  left_fit_average = np.average(left_fit, axis=0)
  right_fit_average = np.average(right_fit, axis=0)
  # create a line based on the average values
  left_line = make_coordinates(image, left_fit_average)
  right_line = make_coordinates(image, right_fit_average)
  return np.array([left_line, right_line])

# python script that does the folowing:
# 1. reads in the image
# 2. converts the image to grayscale
# 3. applies a gaussian blur
# 4. applies the canny edge detector
# 5. applies a mask to the image
# 6. applies the hough transform to the image
# 7. displays the image with the detected lane lines

def output_canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

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


# image = cv.imread('test_image.jpg')

# lane_image = np.copy(image)

# canny_image = output_canny(image)

# cropped_image = region_of_interest(canny_image)

# # 1st param: image, 2nd param: rho, 3rd param: theta, 4th param: threshold, 5th param: placeholder array, 6th param: placeholder array, 7th param: minLineLength, 8th param: maxLineGap
# # threshold: minimum number of votes (intersections in a given grid cell)

empty_array = np.array([])

# lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, empty_array, minLineLength=40, maxLineGap=5)

# average_lines = average_slope_intercept(lane_image, lines)

# line_image = display_lines(lane_image, lines)

# # 1st param: image, 2nd param: alpha, 3rd param: beta, 4th param: gamma
# # 2nd multiply the line image by 0.8
# # 3rd multiply the original image by 1
# # 4th add 1
# combine_images = cv.addWeighted(lane_image, 0.8, line_image, 1, 1)

# cv.imshow('result', combine_images)

# cv.waitKey(0)

cap = cv.VideoCapture("test2.mp4")

while(cap.isOpened()):
  _, frame = cap.read()
  canny_image = output_canny(frame)
  cropped_image = region_of_interest(canny_image)
  lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, empty_array, minLineLength=40, maxLineGap=5)
  average_lines = average_slope_intercept(frame, lines)
  line_image = display_lines(frame, lines)
  combine_images = cv.addWeighted(frame, 0.8, line_image, 1, 1)
  cv.imshow('result', combine_images)
  if cv.waitKey(1) == ord('q'):
    break
    
cap.release()
cv.destroyAllWindows()