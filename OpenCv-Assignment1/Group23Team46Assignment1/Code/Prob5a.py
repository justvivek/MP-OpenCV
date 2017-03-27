# Import numpy and cv2 package
import cv2
# Reading original image from the disk
image= cv2.imread('emusk.png')

# Converting the image from BGR to Grayscale
img= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Displaying the Grayscale image
cv2.imshow('Gray',img)
# Calculating the mean and standard deviation of the grayscale image
mean, sd = cv2.meanStdDev(img)
# Creating the new image img1 by subtracting mean and then dividing by standard deviation from the original image
img1= (img- mean)/sd

# # Displaying the final output image
cv2.imshow('Whitening',img1)

# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()