#importing all the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import cv2
import numpy as np
import os
from os.path import join
import glob


#Getting the images from the designated path and store category names
train_path=os.path.abspath('dataset')
training_names=os.listdir(train_path)
print (training_names)


image_paths=[]
image_classes=[]
rawimage_pix=[]
class_id=0
nbrnames=len(training_names)
labels=[]
color_features=[]


#Fetching files
def get_imgfiles(path):
	all_files=[]
	all_files.extend([join(path,fname)
			for fname in glob.glob(path+"/*")])
	return all_files

#Storing images and their corresponding labels
for training_names,label in zip(training_names,range(nbrnames)):
	class_path=join(train_path,training_names)
	class_files=get_imgfiles(class_path)
	image_paths+=class_files
	labels+=[class_id]*len(class_files)
	class_id+=1
	
#Method 1 for knn:resizes and flattens images and gives a list of raw pixel intensities
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

#Method 2 : Forming color histogram of given images to give to classifier
def extract_color_histogram(image,bins=(8,8,8),size=(32,32)):
	image=cv2.resize(image,size)
	hsv=cv2.cvtColoqr(image,cv2.COLOR_BGR2HSV)
	hist=cv2.calcHist([hsv],[0,1,2],None,bins,[0 , 180, 0, 256, 0, 256])
#inplace normalization
	cv2.normalize(hist,hist)
	return hist.flatten()

for (i,image_path) in enumerate(image_paths):
	image=cv2.imread(image_path)
#extract a color histogram from the image
	raw=image_to_feature_vector(image)
	rawimage_pix.append(raw)

	hist=extract_color_histogram(image)
	color_features.append(hist)

#show update every 5 image
	if i>0 and i% 5==0:
	 	print("[INFO] processed {}/{}".format(i,len(image_paths)))


features=np.array(color_features)
labels=np.array(labels)	 	
rawImages=rawimage_pix

#Automatic grouping of images for training(75/100 ratio) and testing (25/100 ratio)
#m1.normal rawimage pixels
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
#m2.color histogram pixels
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)



#For method 1
model=KNeighborsClassifier(3)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print (acc)
#For method 2
model1 = KNeighborsClassifier(3)
model1.fit(trainFeat, trainLabels)
acc1 = model1.score(testFeat, testLabels)
print (acc1)

