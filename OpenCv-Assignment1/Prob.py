import cv2
import numpy as np

img= cv2.imread('lion.jpg',0)
cv2.imshow('BGR lion', img)


img_smooth1= cv2.GaussianBlur(img, (3,3), 10,20)
img_smooth2= cv2.GaussianBlur(img, (3,3), 150,200)
img_smooth3= cv2.GaussianBlur(img, (5,5), 10,20)
img_smooth4= cv2.GaussianBlur(img, (5,5), 150, 200)


img1= np.concatenate((img,img_smooth1, img_smooth2, img_smooth3, img_smooth4), axis=1)
cv2.imshow('Images using different Kernel and Sigma', img1)
cv2.imwrite('Problem6 Output.jpg', img1)
cv2.waitKeyEx(0)
cv2.destroyAllWindows()

