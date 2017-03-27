# Import cv2 package
import cv2

# Reading original image from the disk
image= cv2.imread('rgb.jpg')
#Displaying the original image
cv2.imshow('RGB', image)

# Converting the image from BGR to Grayscale using the OpenCv inbuilt function
gray_img= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#Displaying the Grayscale image
cv2.imshow('Gray', gray_img)

# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()