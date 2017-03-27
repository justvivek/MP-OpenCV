import cv2

image= cv2.imread('rgb.jpg')
cv2.imshow('RGB', image)

gray_img= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()