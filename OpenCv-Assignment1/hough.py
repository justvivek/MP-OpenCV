import cv2
import numpy as np

img = cv2.imread('road2.jpg')
b,g,r= cv2.split(img)


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
newimg=np.ones(gray.shape,dtype='uint8')*255
newimg[gray<100]=0
cv2.imshow("threshhold image", newimg)

cv2.waitKey(0)

# Apply smoothening to the image(Gaussian Blurr)
smoothing= cv2.GaussianBlur(gray,ksize=(3,3),sigmaX=1,sigmaY=5)
# cv2.Set
cv2.imshow("Smoothened image", smoothing)
edges = cv2.Canny(smoothing, 190 ,250, apertureSize = 3)
# print(edges[204:240,1:40])
cv2.imshow('edges',edges)
# cv2.waitKey(0)
# minLineLength = 1
# maxLineGap = 30
# cv2.HoughLinesP(edges, 1, np.pi/180,100)
lines = cv2.HoughLinesP(edges,1,np.pi/180,70 ,minLineLength=30,maxLineGap=5)
print(lines.shape)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('houghlines5.jpg',img)
cv2.waitKey(0)