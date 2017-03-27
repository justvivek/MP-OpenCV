import cv2

img= cv2.imread('1.jpeg')
cv2.imshow('RGB',img)
img_lab=cv2.cvtColor(img,cv2.COLOR_BGR2Lab)

cv2.imshow('L*a*b*', img_lab)
cv2.waitKey(0)
cv2.destroyAllWindows()