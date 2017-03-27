import cv2

img= cv2.imread('1.jpeg')
cv2.imshow('RGB',img)
img_red=img[:,:,2]
img_green=img[:,:,1]
img_blue=img[:,:,0]

img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img= np.hstack((img_red, img_green))
# img= np.hstack((img, img_blue))
cv2.imwrite("Problem1 Output.jpg", img)
cv2.imshow('Red channel',img_red)
cv2.imshow('Green channel',img_green)
cv2.imshow('Blue channel',img_blue)
cv2.waitKey(0)
cv2.destroyAllWindows()