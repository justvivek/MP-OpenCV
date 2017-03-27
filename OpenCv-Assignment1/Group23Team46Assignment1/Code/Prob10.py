# Import numpy and cv2 package
import cv2
import numpy as np


# Reading original image from the disk as grayscale image
img= cv2.imread('man.jpg',0)

#Smoothening the image using the Gausian Blurr
smooth= cv2.GaussianBlur(img, ksize=(5,5),sigmaX=10,sigmaY=10)
#Applying the Laplacian on the smoothen image
sharp= cv2.Laplacian(smooth,ddepth=cv2.CV_8U, ksize=3)

#Subtracting the sharp image image matrix from the original image matrix
img1= cv2.subtract(img,sharp)

# Displaying original, sharp  and final output images together
img3= np.concatenate((img,sharp,img1),axis=1)

cv2.imshow('Problem10 Output',img3)

# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()