# Import numpy and cv2 package
import cv2
import numpy as np

# Reading original image from the disk as grayscale image
img= cv2.imread('low_illum.jpg',0)

#Smoothening the image using different kernel size and sigmaX & sigmaY
G_smooth1= cv2.GaussianBlur(src=img, ksize=(3,3),sigmaX=3,sigmaY=3)

G_smooth2= cv2.GaussianBlur(src=img, ksize=(3,3),sigmaX=10,sigmaY=10)

G_smooth3= cv2.GaussianBlur(src=img, ksize=(5,5),sigmaX=3,sigmaY=3)

# Displaying the original image and all the output images using different kernel and sigmaX & sigmaY, together
img1= np.concatenate((img,G_smooth1,G_smooth2,G_smooth3), axis=1)
cv2.imshow('Problem6_Output',img1)

# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()