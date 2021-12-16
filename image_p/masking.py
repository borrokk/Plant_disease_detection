# Masking(complete and intermediate), Erosion and Dilation

from PIL import Image
# load the image
image = Image.open('/content/d2.png')
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)

#average of pixel is around 0.5 after normalization, after using median filter

from numpy import asarray
pixels = asarray(image)
# confirm pixel range is 0-255
print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
# convert from integers to floats
pixels = pixels.astype('float32')
# normalize to the range 0-1
pixels /= 255.0
# confirm the normalization
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

import cv2 
import numpy as np

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])
input = cv2.imread('/content/tomato.png')
from google.colab.patches import cv2_imshow 
hsv = cv2.cvtColor(input,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, low_green, high_green)
cv2_imshow(mask)

def detect_leaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # find the brown color
    mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    # find the yellow and green color in the leaf
    mask_yellow_green = cv2.inRange(hsv, (10, 39, 64), (86, 255, 255))
    # find any of the three colors(green or brown or yellow) in the image
    mask = cv2.bitwise_or(mask_yellow_green, mask_brown)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    return res

#final result after masking

image = cv2.imread('/content/tomato.png')
#image.astype(np.float32)
temp = detect_leaf(image) #final result
cv2_imshow(temp)

#Intermediate mask - remove artifacts

def detectingleaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # find the brown color
    mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    # find the yellow and green color in the leaf
    mask_yellow_green = cv2.inRange(hsv, (10, 39, 64), (86, 255, 255))
    # find any of the three colors(green or brown or yellow) in the image
    mask = cv2.bitwise_or(mask_yellow_green, mask_brown)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    return mask

image = cv2.imread('/content/tomato.png')
#image.astype(np.float32)
temp = detectingleaf(image) #Intermediate mask - remove artifacts
cv2_imshow(temp)

#erosion and dilation
import cv2
import numpy as np
 
# Reading the input image
img = cv2.imread('/content/er_d.png', 0)
 
# Taking a matrix of size 5 as the kernel
kernel = np.ones((5,5), np.uint8)
 
# The first parameter is the original image, kernel is the matrix with which image is
# convolved and third parameter is the number of iterations, which will determine how much
# you want to erode/dilate a given image.
img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)
 
cv2_imshow(img)
cv2_imshow(img_erosion)
cv2_imshow(img_dilation)

#colour and shadow mask function

def leaf(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # find the brown color
  mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    # find the yellow and green color in the leaf
  mask_yellow_green = cv2.inRange(hsv, (10, 39, 64), (86, 255, 255))
  return mask_yellow_green

# done to get shadow mask
def dis(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # find the brown color
  mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
  # finf green
  mask_yellow_green = cv2.inRange(hsv, (10, 39, 64), (86, 255, 255))
  mask = cv2.bitwise_xor(mask_yellow_green, mask_brown)
  mask = cv2.bitwise_not(mask)
  return mask
  #bitwise_not done for inverting mask

#plotting
image = cv2.imread('/content/tomato.png')
img = cv2.imread('/content/tomato.png')
temp = leaf(image) #color mask
temp2 = dis(img) #shadow mask
cv2_imshow(temp)
cv2_imshow(temp2)

