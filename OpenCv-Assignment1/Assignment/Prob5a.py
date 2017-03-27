import cv2

image= cv2.imread('emusk.png')
img= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray',img)
mean, sd = cv2.meanStdDev(img)
img1= (img- mean)/sd
cv2.imshow('Whitening',img1)
cv2.waitKey(0)