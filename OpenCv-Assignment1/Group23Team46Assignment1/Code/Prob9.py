# Import numpy and cv2 package
import cv2
import numpy as np

#create a white image(square box) of dimension 250X250
img= np.ones((250,250), dtype=np.uint8)*255

#Make square boxes of 50X50
cv2.line(img,(0,50),(250,50),0,1)
cv2.line(img,(0,100),(250,100),0,1)
cv2.line(img,(0,150),(250,150),0,1)
cv2.line(img,(0,200),(250,200),0,1)

cv2.line(img,(50,0),(50,250),0,1)
cv2.line(img,(100,0),(100,250),0,1)
cv2.line(img,(150,0),(150,250),0,1)
cv2.line(img,(200,0),(200,250),0,1)

#draw lines of 45 degrees on lower half of square
cv2.line(img,(0,0),(250,250),0,1)
cv2.line(img,(0,50),(200,250),0,1)
cv2.line(img,(0,100),(150,250),0,1)
cv2.line(img,(0,150),(100,250),0,1)

#Display the original image
cv2.imshow('Image',img)

# kernel we are using to detect 45 degree lines
kernel = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])/12

# Convolving the image with the kernel using the filter2d() function of the OpenCV library
_45degreelines = cv2.filter2D(src=img,ddepth=cv2.CV_8U,kernel=kernel)

#Displaying the image in which we have detected the 45 degrees lines only
cv2.imshow('Output Image with 45 degree lines only ',_45degreelines )

# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()
