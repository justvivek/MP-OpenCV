# Import numpy and cv2 package
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

def calculate_meanshift(image):
    # Reading original image from the disk
    img= cv2.imread(image)
    #Displaying the original image
    # cv2.imshow('RGB',img)
    # print(img.shape)
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    bandwidth = estimate_bandwidth(vectorized, quantile=0.1, n_samples=100)

    ms= MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(vectorized)

    labels = ms.labels_
    # print(labels.shape)
    # print(type(labels))
    cluster_centers = ms.cluster_centers_
    cluster_centers = np.uint8(cluster_centers)
    # print(cluster_centers.shape)
    # print(type(cluster_centers))
    # labels_unique = np.unique(labels)
    res= cluster_centers[labels]
    # print(res.shape)
    segmented_image= res.reshape((img.shape))
    return segmented_image
    # segmented_image= np.reshape(( img.shape))
    # cv2.imshow('Segmented Image', segmented_image)


result = calculate_meanshift('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg1.jpg')
cv2.imshow('Mean Shift Seg1 Image', result)

result = calculate_meanshift('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg2.jpg')
cv2.imshow('Mean Shift Seg2 Image', result)

result = calculate_meanshift('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg3.jpg')
cv2.imshow('Mean Shift Seg3 Image', result)

result = calculate_meanshift('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg4.jpg')
cv2.imshow('Mean Shift Seg4 Image', result)

result = calculate_meanshift('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg5.jpg')
cv2.imshow('Mean Shift Seg5 Image', result)



# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()

'''
Meanshift looks very similar to K-Means, they both move the point closer to the cluster centroids. One may wonder:
How is this different from K-Means? K-Means is faster in terms of runtime complexity!
The key difference is that Meanshift does not require the user to specify the number of clusters. In some cases,
it is not straightforward to guess the right number of clusters to use. In K-Means, the output may end up having too
few clusters or too many clusters to be useful.
At the cost of larger time complexity, Meanshift determines the number of clusters suitable to the dataset provided.
'''