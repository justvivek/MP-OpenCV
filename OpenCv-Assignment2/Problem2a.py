# Import numpy and cv2 package
import cv2
import numpy as np
# Importing the library which classifies set of observations into clusters
from scipy.cluster import vq
from sklearn.cluster import MeanShift, estimate_bandwidth

#Segmenting images using the meanShift inbuilt library from sklearn
#here we does not require us to specify the number of clusters.
def calculate_meanshift(image):
    img= cv2.imread(image)
    #flattening the image and converting to array of floats
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    # Compute clustering with MeanShift
    # bandwidth can be automatically detected using estimate_bandwidth
    bandwidth = estimate_bandwidth(vectorized, quantile=0.1, n_samples=100)
    ms= MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # Perform clustering Samples to cluster.
    ms.fit(vectorized)
    # Labels of each point.
    labels = ms.labels_
    # Coordinates of cluster centers.
    cluster_centers = ms.cluster_centers_
    cluster_centers = np.uint8(cluster_centers)
    # Assigning each label to one of the clusters
    res= cluster_centers[labels]
    #rebuilding the segmented image from flatten array
    segmented_image= res.reshape((img.shape))
    return segmented_image

#Segmenting images using the kmeans inbuilt library from scipy
# Here we have to explicitly mentions the cluster (5)
def find_kmean(image, k=5):
    img= cv2.imread(image)
    # flattening the image and converting to array of floats
    vectorized= img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    # Performs k-means on a set of observation vectors forming k(5) clusters. This yields a code book mapping centroids to codes
    center, dist = vq.kmeans(vectorized,k)
    # Converting back to unsigned int
    center = np.uint8(center)
    # Assign codes from a code book to observations.
    # and get length N array holding the code book index for each observation and distance between the observation and its nearest code.
    code, distance = vq.vq(vectorized, center)
    ## Assigning each label to one of the clusters
    res= center[code]
    # rebuilding the segmented image from flatten array
    f_res= res.reshape((img.shape))
    return f_res

# Find the segmented image  using meanshift  of the image calling calculate_meanshift function and show the image
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

# Find the Kmean cluster of the image calling find_kmean function and show the image
result = find_kmean('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg1.jpg')
cv2.imshow('k=4 kMeans Seg1 image', result)

result = find_kmean('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg2.jpg')
cv2.imshow('k=4 kMeans Seg2 image', result)

result = find_kmean('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg3.jpg')
cv2.imshow('k=4 kMeans Seg3 image', result)

result = find_kmean('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg4.jpg')
cv2.imshow('k=4 kMeans Seg4 image', result)

result = find_kmean('/home/vivek/PycharmProjects/Assignment2/Input_Images/Seg5.jpg')
cv2.imshow('k=4 kMeans Seg5 image', result)


# Display the images and wait till any key is pressed
cv2.waitKey(0)
# Destroy all the windows created by the imshow() function of the OpenCV
cv2.destroyAllWindows()