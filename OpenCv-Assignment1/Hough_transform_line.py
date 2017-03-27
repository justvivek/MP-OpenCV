import cv2
import numpy as np

img = cv2.imread('sudoku.png')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges= cv2.Canny(gray,100,150, apertureSize=3)

cv2.imshow('grayscale',gray)
# eqalized =cv2.equalizeHist(gray)
# cv2.imshow('Equalized',eqalized)
# blur= cv2.GaussianBlur(eqalized,ksize=(5,5),sigmaX=20,sigmaY=20)
# cv2.imshow('blurred',blur)
# edges1= cv2.Canny(eqalized,100,150, apertureSize=3 )
# cv2.imshow('Edges1', edges1)

#---------------------------------------------------------------------------------#

minLineLength =10
maxLineGap = 5
# cv2.HoughLinesP(edges,)
lines = cv2.HoughLinesP(edges,1,np.pi/180,200,minLineLength,maxLineGap)
for line in lines[:]:
    for x1,y1,x2,y2 in line[0]:
        cv2.line(gray,(x1,y1),(x2,y2),(0,0,255),1)

cv2.imshow('lines122', gray)
# cv2.imwrite('houghlines5.jpg',img)




#---------------------------------------------------------------------------------#
# cv2.imshow('Original',img)
#cv2.imshow('GrayScale',gray)
# cv2.imshow('Edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()