import cv2
import numpy as np


original  = cv2.imread('road2.jpg')
img  = cv2.imread('road2.jpg')

b,g,r = cv2.split(img)

r[r<235]=0
g[g<235]=0
masked_img = np.zeros(img.shape,dtype='uint8')
masked_img[:,:,2]=r
b_temp = b/2
masked_img[:,:,0]= b_temp.astype('uint8')
masked_img[:,:,1]=g
cv2.imshow('masked img', masked_img)
gray = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
bw = np.ones(gray.shape,dtype='uint8')*255
bw[gray<200]=0
cv2.imshow('res_img',bw)

lines = cv2.HoughLinesP(bw,1,np.pi/180,80,minLineLength=2,maxLineGap=15)
for line in lines[:]:
     for x1,y1,x2,y2 in line:
         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('res',img)
cv2.imwrite('Problem11 Output.jpg',np.concatenate((original,img),axis=1))
cv2.waitKey(0)
cv2.destroyAllWindows()