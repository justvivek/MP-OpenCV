import cv2
import numpy as np

img = cv2.imread('lion.jpg',0)

#---------------------------------------------------------------------------------#

# creating noise matrix
rows,cols = img.shape;
noise = np.random.randn(rows,cols)*10
noise = noise.astype(dtype=np.int8)
img_new = cv2.add(img,noise,dtype=cv2.CV_8U)
cv2.imshow('original',img)
cv2.imshow('new_img',img_new)

res_img1=cv2.GaussianBlur(src=img,ksize=(5,5),sigmaX=10,sigmaY=10)
res_img2=cv2.GaussianBlur(src=img_new,ksize=(5,5),sigmaX=10,sigmaY=10)

cv2.imshow('res1',res_img1)
cv2.imshow('res2',res_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()