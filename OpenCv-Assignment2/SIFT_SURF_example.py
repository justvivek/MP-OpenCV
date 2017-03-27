import cv2
import numpy as np

'''
There are many, many alternatives to SIFT and SURF. ORB is one. BRIEF, BRISK, FREAK, KAZE, and AKAZE are others.

'''

img= cv2.imread('/home/vivek/PycharmProjects/Assignment2/Input_Images/lena.jpg')
cv2.imshow('RGB', img)
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape)
cv2.imshow('Gray', gray)


sift = cv2.xfeatures2d.SIFT_create()
cv2.xfeatures2d.BriefDescriptorExtractor_create()
(kps, descs)= sift.detectAndCompute(gray,None)
print("#kps: {}, descriptors: {}  ".format(len(kps), descs.shape))


surf = cv2.xfeatures2d.SURF_create()
(kps, descs)= surf.detectAndCompute(gray, None)
print("#kps : {}, descriptors: {}".format(len(kps), descs.shape))

# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()