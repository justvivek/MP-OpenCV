# Import numpy and cv2 package
import cv2
import numpy as np

# Reading original image from the disk
original  = cv2.imread('road2.jpg')
img  = cv2.imread('road2.jpg')

#Split the image into r,g,b channel using inbuilt function of OpenCv
b,g,r = cv2.split(img)

#thresholding the red and green channel where pixel intensity  is less than 235 make it 0.
r[r<235]=0
g[g<235]=0
#Creating a new image of same size as of original image
masked_img = np.zeros(img.shape,dtype='uint8')
#assigining the new values of r,g,b channel of original image to masked_image
masked_img[:,:,2]=r
b_temp = b/2
masked_img[:,:,0]= b_temp.astype('uint8')
masked_img[:,:,1]=g
#Displaying the masked_image
cv2.imshow('masked img', masked_img)

#Converting the masked_img to grayscale using inbuilt function CV
gray = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
#Creating new image 'thresh'of size of gray image
thresh = np.ones(gray.shape,dtype='uint8')*255
#thresholding the new image 'thresh' where pixel intensity is less than 200 make it zer0
thresh[gray<200]=0

# Applying the inbuilt HoughlinesP function of the OpenCV to find coordinates of the end points of line
# in Hough space
lines = cv2.HoughLinesP(thresh,1,np.pi/180,80,minLineLength=2,maxLineGap=15)
# We got the coordinates of each line which are passing the threshold limit of votes set by us i.e 80
# using those coordinates we are drawing lines on the original image which will highlight all the lanes in the road
# with green color
for line in lines[:]:
     for x1,y1,x2,y2 in line:
         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# # Displaying the final output image
cv2.imshow('res',img)
# Displaying the original image and final output image together
cv2.imshow('Problem11 Output',np.concatenate((original,img),axis=1))

# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()