import cv2
import numpy as np

# img= cv2.imread('Stars-Night-Sky.jpg',0)
img= cv2.imread('1.png',0)
# img= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', img)

# cany= cv2.Canny(img, 127,200,apertureSize=3)
# cv2.imshow('Canny',cany)

smooth=cv2.GaussianBlur(img, ksize=(3,3),sigmaX=5,sigmaY=5)
log1= cv2.Laplacian(img, ddepth=cv2.CV_8U,ksize=3)
r1= len(log1)# Simply returns the no of rows in log1 image
# print('no of stars in original',r1)
log2= cv2.Laplacian(smooth, ddepth=cv2.CV_8U,ksize=3)
# detector= cv2.SimpleBlobDetector_Params()

# keypoints= detector
# print(len(keypoints))

r2= len(log2)

# print('NO of stars after smoothening',r2)
# cv2.Laplacian()

dup= np.zeros(log2.shape, dtype=np.uint8)
dup[log2>127]=1
print(sum(sum(dup)))

cv2.imshow('Smooth', smooth)
cv2.imshow('log1',log1)
cv2.imshow('log2',log2)

cv2.waitKey(0)
cv2.destroyAllWindows()