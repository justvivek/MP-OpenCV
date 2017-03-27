import cv2
import numpy as np

img = cv2.imread('Coins_gray.png',0)
cv2.imshow('Original',img)
#print(img)

row,col= img.shape
noise= np.random.randn(row,col)*70
noise= noise.astype(dtype=np.int8)

#print(noise)

salt_pepper_noise= img[:,:]
salt_pepper_noise[noise>110]=255
salt_pepper_noise[noise<-110]=0
cv2.imshow('Salt_pepper_noise', salt_pepper_noise)

# Median filtering applied to image having salt and pepper noise(it's the most efeective one in case of salt n pepper)
median= cv2.medianBlur(salt_pepper_noise, 3)
cv2.imshow('Median_Smoothening', median)

cv2.waitKey(0)
cv2.destroyAllWindows()
