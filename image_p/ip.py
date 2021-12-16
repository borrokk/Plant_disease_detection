#image processing operations

# rgb to lab space conversion
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib
img = cv2.imread('/content/drive/MyDrive/dt/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG')
img1 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) 
plt.imsave('name.png', img1)

import cv2
input = cv2.imread('/content/tomato.png')
from google.colab.patches import cv2_imshow 
cv2_imshow(input)
cv2.waitKey(0)
cv2.destroyAllWindows()

lab = cv2.cvtColor(input,cv2.COLOR_BGR2LAB)
cv2_imshow(lab)

L,A,B=cv2.split(lab)
cv2_imshow(L) # For L Channel
cv2_imshow(A) # For A Channel 
cv2_imshow(B) # For B Channel

cv2.waitKey(0)
cv2.destroyAllWindows()

#L channel adjustment with - contrast
#applying contrast
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(L)
cv2_imshow(cl)

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,A,B))
cv2_imshow(limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2_imshow(final)

#Applying filter to L channel

bilateral_filter = cv2.bilateralFilter(src=L, d=3, sigmaColor=30, sigmaSpace=30) #know what each filter and arg means
cv2_imshow(L)

cv2_imshow(bilateral_filter)

cv2.waitKey(0)
cv2.destroyAllWindows()

#blurring L channel

median = cv2.medianBlur(src=L, ksize=1)

cv2_imshow(L)
cv2_imshow( median)

img = cv2.merge((median,A,B)) 
cv2_imshow(img)     
lab = cv2.cvtColor(img,cv2.COLOR_LAB2RGB)
cv2_imshow(lab)

#Applying blur to a,b channel and merging to get reconstructed image

# reconstructing
med = cv2.medianBlur(src=A, ksize=1)
medB = cv2.medianBlur(src= B, ksize=1)
cv2_imshow(A)
cv2_imshow(B)
cv2_imshow(L)
cv2_imshow(med)
cv2_imshow(medB)
img1 = cv2.merge((bilateral_filter,med,medB))
cv2_imshow(img1)     
lab = cv2.cvtColor(img1,cv2.COLOR_LAB2BGR)
cv2_imshow(lab) # final reconstructed image

from PIL import Image, ImageStat
im = Image.open('/content/d2.png')
stat = ImageStat.Stat(im)
print(stat.median)
#print(pixel)

from PIL import Image, ImageStat
im = Image.open('/content/tomato.png')
stat = ImageStat.Stat(im)
print(stat.median)

#if pixel value not close to green channel, more processing is needed

#adding gaussian blur 

gb = cv2.GaussianBlur(L, (3,3), 1,1)
gb1 = cv2.GaussianBlur(A, (3,3), 1,1)
gb2 = cv2.GaussianBlur(B, (3,3), 1,1)
cv2_imshow(gb)
cv2_imshow(gb1)
cv2_imshow(gb2)
