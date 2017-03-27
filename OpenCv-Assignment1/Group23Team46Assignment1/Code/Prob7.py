# Import numpy and cv2 package
import cv2
import numpy as np

# Reading original image from the disk as grayscale image
img = cv2.imread('Coins_gray.png',0)
#Displaying the image
cv2.imshow('Original',img)

# creating noise matrix
row,col= img.shape
noise= np.random.randn(row,col)*70
noise= noise.astype(dtype=np.int8)

# adding Salt and Pepper noise to the image
salt_pepper_noise= img[:,:]
salt_pepper_noise[noise>110]=255
salt_pepper_noise[noise<-110]=0

#Displaying the image with salt an pepper noise
cv2.imshow('Salt_pepper_noise', salt_pepper_noise)


# Median filtering applied to image having salt and pepper noise(it's the most efeective one in case of salt n pepper)
median= cv2.medianBlur(salt_pepper_noise, 3)
cv2.imshow('Median_Smoothening', median)


# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()
