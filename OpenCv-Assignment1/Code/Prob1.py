# Import numpy and cv2 package
import cv2
import numpy as np

# Reading original image from the disk
img= cv2.imread('1.jpeg')
#Displaying the original image
cv2.imshow('RGB',img)

# Separate the red,blue and green channel image from the original 'img'
img_red=img[:,:,2]
img_green=img[:,:,1]
img_blue=img[:,:,0]

# Displaying the red,green and blue channel output image together
cv2.imshow('Problem1 Output.jpg', np.concatenate((img_red,img_green,img_blue),axis=1))
# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()