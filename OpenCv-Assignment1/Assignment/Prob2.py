import cv2

img= cv2.imread('1.jpeg')
cv2.imshow('RGB',img)
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img_hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

variance=img_hsv[:,:,2]
saturation=img_hsv[:,:,1]
hue=img_hsv[:,:,0]

cv2.imshow('HSV',img_hsv)
cv2.imshow('HLS',img_hls)

cv2.imshow('Hue',hue)
cv2.imshow('Saturation',saturation)
cv2.imshow('Variance', variance)

cv2.waitKey(0)
cv2.destroyAllWindows()