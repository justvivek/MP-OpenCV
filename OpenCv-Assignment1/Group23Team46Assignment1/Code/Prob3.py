# Import cv2 package
import cv2

# Reading original image from the disk
img= cv2.imread('1.jpeg')
#Displaying the original image
cv2.imshow('RGB',img)
#Converting the original image from BGR to L*a*b* using OpenCv inbuilt function
img_lab=cv2.cvtColor(img,cv2.COLOR_BGR2Lab)

#Displaying the L*a*b* image using OpenCv inbuilt function
cv2.imshow('L*a*b*', img_lab)
# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()