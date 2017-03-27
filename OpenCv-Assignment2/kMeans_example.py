# Import numpy and cv2 package
import cv2
import numpy as np
from scipy.cluster import vq
# Reading original image from the disk

def find_kmean(image, k=5):
    img= cv2.imread(image)
    print(img.shape)
    #Displaying the original image
    # cv2.imshow('RGB',img)
    vectorized= img.reshape((-1,3))
    print(vectorized.shape)
    vectorized = np.float32(vectorized)
    # k=6
    center, dist = vq.kmeans(vectorized,k)
    # print(type(center))
    center = np.uint8(center)
    code, distance = vq.vq(vectorized, center)
    res= center[code]
    print(res.shape)
    f_res= res.reshape((img.shape))
    return f_res

result = find_kmean('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg1.jpg')
cv2.imshow('k=4 Seg1 image', result)

result = find_kmean('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg2.jpg')
cv2.imshow('k=4 Seg2 image', result)

result = find_kmean('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg3.jpg')
cv2.imshow('k=4 Seg3 image', result)

result = find_kmean('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg4.jpg')
cv2.imshow('k=4 Seg4 image', result)

result = find_kmean('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg5.jpg')
cv2.imshow('k=4 Seg5 image', result)
# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()




'''

# Import numpy and cv2 package
import cv2
import numpy as np

# Reading original image from the disk
img= cv2.imread('1.jpeg')
#Displaying the original image
cv2.imshow('RGB',img)



# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()

'''