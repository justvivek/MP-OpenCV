
import cv2
import numpy as np


original  = cv2.imread('road1.png')
# new_img= cv2.resize(original,(300,300), fx=10,fy=10)
# cv2.imshow('new image', new_img)
gray  = cv2.imread('road1.png',0)
#
# b,g,r = cv2.split(img)
# # img1= np.concatenate((r,g,b), axis=1)
# # cv2.imshow('RGB', img1)
#
# r[r<235]=0
# g[g<235]=0
# # img1= np.concatenate((r,g,b), axis=1)
# # cv2.imshow('RGB1', img1)
# masked_img = np.zeros(img.shape,dtype='uint8')
# masked_img[:,:,2]=r
# b_temp = b/2
# masked_img[:,:,0]= b_temp.astype('uint8')
# masked_img[:,:,1]=g
# cv2.imshow('masked img', masked_img)
# gray = cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray', gray)
bw = np.ones(gray.shape,dtype='uint8')*255
bw[gray<200]=0
# cv2.imshow('res_img_before_smooth',bw)
# cv2.GaussianBlur(bw, (5,5),sigmaX=3,sigmaY=3)
cv2.imshow('res_img',bw)

# cv2.imshow('masked',masked_img)
# cv2.imshow('original',original)
# gray_img= cv2.cvtColor(masked_img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray_img',gray_img)
# bw = cv2.Canny(image=gray_img,threshold1=100,threshold2=255)
#

lines = cv2.HoughLinesP(bw,1,np.pi/180,70,minLineLength=2,maxLineGap=10)
for line in lines[:]:
     for x1,y1,x2,y2 in line:
         cv2.line(gray,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('res',gray)
# cv2.imwrite('Ques11_b.png',np.hstack((original,img)))
cv2.waitKey(0)
cv2.destroyAllWindows()