import cv2
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from scipy.cluster import vq
import os
from os.path import join
import glob

# Load the classifier, class names, scaler, number of clusters and vocabulary
samples = np.loadtxt('samples.data',np.float32)
responses = np.loadtxt('responses.data',np.float32)
classes_names = np.load('training_names.data.npy')
voc = np.load('voc.data.npy')
k = 50  # Loading the number of cluster

# Training the Knearest classifier with the test descriptors
clf = neighbors.KNeighborsClassifier()
clf.fit(samples,responses)  # Train model using the training samples and corresponding responses

# Get the path of the testing image(s) and store them in a list
test_path = os.path.abspath('/home/vivek/PycharmProjects/Assignment2/Input_Images/Test_Images')
# Get the training classes names and store them in a list


image_paths=[]
testing_names= os.listdir(test_path)
#Fetching files
def get_imgfiles(path):
	all_files=[]
	all_files.extend([join(path,fname)
			for fname in glob.glob(path+"/*")])
	return all_files

for testing_name in testing_names:
    dir = join(test_path,testing_name)
    class_path = get_imgfiles(dir)
    image_paths +=class_path

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

# Computing the histogram of features
test_features = np.zeros((len(image_paths), k), np.float32)
for i in range(len(image_paths)):
    words, distance = vq.vq(descriptor_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1  # Calculating the histogram of features


# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)  # Getting the number of occurrences of each word
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
# Assigning weight to one that is occurring more frequently


# Perform the predictions
results= clf.predict(test_features)

if results[0][0] == 0:  # results[0][0] will have the predicted class
    prediction = "Horse"
else:
    prediction = "Bike"

accuracy = clf.score(results, responses)
print(accuracy) # we are getting 89% as we increase the no og clusters our accuracy increases to 94%