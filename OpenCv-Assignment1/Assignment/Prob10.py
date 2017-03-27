import cv2
import numpy as np

img= cv2.imread('man.jpg',0)
smooth= cv2.GaussianBlur(img, ksize=(5,5),sigmaX=10,sigmaY=10)
sharp= cv2.Laplacian(smooth,ddepth=cv2.CV_8U, ksize=3)

img1= cv2.subtract(img,sharp)

img3= np.concatenate((img,sharp,img1),axis=1)

cv2.imshow('Problem10 Output',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()