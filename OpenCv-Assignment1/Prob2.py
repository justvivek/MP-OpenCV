# Import    cv2 package
import cv2


# Reading original image from the disk
img= cv2.imread('1.jpeg')
#Displaying the original image
cv2.imshow('RGB',img)
#Converting the original image from BGR to HSV and HLS using OpenCv inbuilt function
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img_hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

#Extracting out the variance, saturation, Hue from img_hsv image
variance=img_hsv[:,:,2]
saturation=img_hsv[:,:,1]
hue=img_hsv[:,:,0]

#Displaying the HSV and HLS image
cv2.imshow('HSV',img_hsv)
cv2.imshow('HLS',img_hls)

#Displaying separately the Hue ,Saturation and Variance Channel images
cv2.imshow('Hue',hue)
cv2.imshow('Saturation',saturation)
cv2.imshow('Variance', variance)

# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()