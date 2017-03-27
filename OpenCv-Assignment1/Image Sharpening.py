import cv2
import numpy as np

img= cv2.imread('man.jpg',0)
smooth= cv2.GaussianBlur(img, ksize=(5,5),sigmaX=100,sigmaY=100)
sharp= cv2.Laplacian(smooth,ddepth=cv2.CV_8U, ksize=3)
# smooth1=cv2.GaussianBlur(sharp, ksize=(3,3),sigmaX=3,sigmaY=3)

# laplacian= np.array(([0,1,0],[1,-4,1],[0,1,0]), dtype='int')*2
# img2= cv2.filter2D(img, ddepth=cv2.CV_8U,kernel=laplacian)
# sharp= sharp.astype(dtype= np.int8)
img1= cv2.subtract(img,sharp)
# img1= img+smooth1
# img1= img1.astype(dtype=np.int8)

img3= np.concatenate((img,smooth,sharp,img1),axis=1)

cv2.imshow('image',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()