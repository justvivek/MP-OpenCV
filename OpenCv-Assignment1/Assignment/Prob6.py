import cv2
import numpy as np

img= cv2.imread('low_illum.jpg',0)

G_smooth1= cv2.GaussianBlur(src=img, ksize=(3,3),sigmaX=3,sigmaY=3)

G_smooth2= cv2.GaussianBlur(src=img, ksize=(3,3),sigmaX=10,sigmaY=10)

G_smooth3= cv2.GaussianBlur(src=img, ksize=(5,5),sigmaX=3,sigmaY=3)

img1= np.concatenate((img,G_smooth1,G_smooth2,G_smooth3), axis=1)
cv2.imshow('Problem6_Output',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()