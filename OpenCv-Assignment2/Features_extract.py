import cv2
import numpy as np
from scipy.cluster import vq
import os
from os.path import join
import glob
from sklearn.preprocessing import StandardScaler # Importing the library that supports centering and scaling vectors

# Get the path of the training set
train_path = os.path.abspath('/home/vivek/PycharmProjects/Assignment2/Input_Images/Training_Images')
# Get the training classes names and store them in a list
training_names= os.listdir(train_path)
print(training_names)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []  # Inilialising the list
image_classes = []  # Inilialising the list
class_id = 0

#Fetching files
def get_imgfiles(path):
	all_files=[]
	all_files.extend([join(path,fname)
			for fname in glob.glob(path+"/*")])
	return all_files


for training_name in training_names:
    dir = join(train_path, training_name)
    class_path = get_imgfiles(dir)
    image_paths +=class_path
    image_classes +=[class_id]*len(class_path)
    class_id +=1

feature_detector = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
descriptor_list = []
# Reading the image and calculating the features and corresponding descriptors
for image_path in image_paths:
    img= cv2.imread(image_path)
    (kps, descs) = feature_detector.detectAndCompute(img, None) # Computing the key points and the descriptors
    descriptor_list.append((image_path,descs))  # Appending all the descriptors into the single list

# Stack all the descriptors vertically in a numpy array
descriptors = descriptor_list[0][1]
for image_path, descriptor in descriptor_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

#Perform k-means clustering
k=500
center, _ = vq.kmeans(descriptors,k,1)

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), k), np.float32 )
for i in range(len(image_paths)):
    code, dist = vq.vq(descriptors, center)
    for c in code:
        im_features[i][c] +=1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
# Calculating the number of occurrences
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
# Giving weight to one that occurs more frequently

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)  # Scaling the visual words for better Prediction

# Saving the contents into a file
np.savetxt("samples.data",im_features)
np.savetxt("responses.data",np.array(image_classes))
np.save("training_names.data",training_names)
np.save("stdSlr.data",stdSlr)
np.save("voc.data",center)