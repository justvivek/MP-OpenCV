# Import numpy and cv2 package
import cv2
import numpy as np

# Creating a binary image
img= np.zeros((250,250), dtype=np.uint8)
subimg= img[75:175, 75:175]
subimg[:,:]= 255
#Dispalying the image
cv2.imshow('Original',img)

## Applying the SOBEL OPERATOR

# If our Output datatype is cv2.CV_8U or np.uint8 then there is a slight problem with that.
# Black-to-White transition is taken as Positive slope (it has a positive value)
# while White-to-Black transition is taken as a Negative slope (It has negative value).
# So when you convert data to np.uint8, all negative slopes are made zero, in that case we miss that edge.

# If we want to detect both edges, better option is to keep the output datatype to some
# higher forms, like cv2.CV_16S, cv2.CV_64F etc, take its absolute value and then convert back to cv2.CV_8U.

# soblex8u= cv2.Sobel(img, cv2.CV_8U,1,0,ksize=5)
sobelx64f= cv2.Sobel(img, cv2.CV_64F,1,0,ksize=5)
# cv2.imshow('Sobelx8U',soblex8u)
abs_sobel64f= np.absolute(sobelx64f)
sobelx_8u = np.uint8(abs_sobel64f)

sobely64f= cv2.Sobel(img, cv2.CV_64F,0,1,ksize=5)
# cv2.imshow('Sobelx8U',soblex8u)
abs_sobel64f= np.absolute(sobely64f)
sobely_8u = np.uint8(abs_sobel64f)

img1= np.concatenate((img, sobelx64f, sobely64f), axis=1)
img2= np.concatenate((img, sobelx_8u, sobely_8u), axis=1)


cv2.imwrite('Problem8 Output Detecting Single edges.jpg', img1)
cv2.imwrite('Problem8 Output Detecting Both edges.jpg', img2)


## Applying the PREWITT filter

kernely= np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely= kernely.astype(dtype=np.int8)
prewitt_y= cv2.filter2D(img,-1,kernely)
cv2.imshow('prewitty',prewitt_y)

kernelx= np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
kernelx= kernelx.astype(dtype=np.int8)
prewitt_x= cv2.filter2D(img,-1,kernelx)
cv2.imshow('prewittx',prewitt_x)

# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()