import cv2
import numpy as np

img= np.ones((250,250), dtype=np.uint8)*255

cv2.line(img,(0,50),(250,50),0,1)
cv2.line(img,(0,100),(250,100),0,1)
cv2.line(img,(0,150),(250,150),0,1)
cv2.line(img,(0,200),(250,200),0,1)

cv2.line(img,(50,0),(50,250),0,1)
cv2.line(img,(100,0),(100,250),0,1)
cv2.line(img,(150,0),(150,250),0,1)
cv2.line(img,(200,0),(200,250),0,1)

cv2.line(img,(250,0),(0,250),0,1)
cv2.line(img,(0,0),(250,250),0,2)
cv2.line(img,(0,50),(200,250),0,1)
cv2.line(img,(0,100),(150,250),0,1)
cv2.line(img,(0,150),(100,250),0,1)

cv2.imshow('Image',img)

# kernel we are using to detect 45 degree lines
kernel = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])/12
_45degreelines = cv2.filter2D(src=img,ddepth=cv2.CV_8U,kernel=kernel)

cv2.imshow('Output Image with 45 degree lines only ',_45degreelines )

cv2.waitKey(0)
cv2.destroyAllWindows()
