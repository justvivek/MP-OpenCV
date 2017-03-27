# Import numpy and cv2 package
import cv2
import numpy as np

# Reading original image from the disk
temp=cv2.imread('valley.jpg')
# Converting the image from BGR to Grayscale using the OpenCv inbuilt function
img= cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
#Displaying the Grayscale image
cv2.imshow('Gray Scale Image', img)
#Getting the size of the image in row and col variables(no of rows and no of columns)
row,col=img.shape

##Equalization by calculating PDF and CDF explicitly without using inbuilt function

#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
hist= cv2.calcHist([img],[0],None,[256],[0,256])
hist1= hist/(row*col) # Probability of each pixel intensity(PDF of pixel intesity)
cdf= hist1.cumsum() ## cummulative sum of probabilty distributive function

#Normalizing the CDF calculated above
cdf_normalized = cdf * hist.max()/ cdf.max()
#Masking the cdf calculated above
cdf_m=np.ma.masked_equal(cdf,0)
cdf_m=(cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf=np.ma.filled(cdf_m,0).astype('uint8')

# Contructing histogram equalized image
hist_e=cdf[img]
#Displaying the Histogram Equalized Image
cv2.imshow('Histogram Equalized Image', hist_e)


#Equalization using CV inbuilt function for histogram equalization
hist_equalized= cv2.equalizeHist(img)


## Displaying all the output images together
img1=np.concatenate((img, hist_e,hist_equalized), axis=1)
cv2.imshow('Problem5b Output', img1)

# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()