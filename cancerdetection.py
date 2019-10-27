"""
This project was created and being maintained by Shobhit Agarwal. The model created in this program CAN NOT be used in transfer learning.

Thank you and enjoy reverse engineering this code :)

PERMISSION NEEDED TO USE AND DISTRIBUTE CODE
PROPERTY OF APPARATUS DIAGNOSING
For permission, email: shobhitagarwal122@gmail.com or dev.shobhitagarwal@gmail.com


WORK IN PROGRESS...

"""


from glob import glob
from cv2 import cv2
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import fnmatch
from progressbar import printProgressBar
import time 
import numpy as np

imagePatches = glob('/mnt/c/Users/shobh/Documents/breast-histopathology-images/IDC_regular_ps50_idx5/**/*.png', recursive=True)



pix_array = []

# for filename in imagePatches:
#     #print(filename)
#     im_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#     plt.imshow(im_array, cmap="gray")
#     plt.show()
#     break


""""OPTIMIZTION PROBLEM: FIGURE OUT HOW TO EFFECTIVELY CYCLE THROUGH ALL IMAGES"""



trainingData = []

CATEGORIES = ["BENIGN", "MALIGNANT"]




#Initialize crazy function 
def create_training_data():
    patternZero = '*class0.png'
    patternOne = '*class1.png'
    classZero = fnmatch.filter(imagePatches, patternZero)
    l1 = len(classZero[0:1000])
    classOne = fnmatch.filter(imagePatches, patternOne)
    l2 = len(classOne[0:1000])
    # Update: module needs to have two lengths. These categories are for the different images (Benign and Malignant)
    for i, filename in enumerate(classZero[0:1000]):
        #print(filename)
        try:
            
            im_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(im_array, (50, 50))
            trainingData.append([new_array, 0])
            time.sleep(0.1)
            printProgressBar(i + 1, l1, prefix = 'Progress:', suffix = 'Complete', length = 50)  
        except Exception as e:
            print(e)
            pass
    for i, filename in enumerate(classOne[0:1000]):
        try:
            im_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(im_array, (50, 50))
            trainingData.append([new_array, 1])
            time.sleep(0.1)
            printProgressBar(i + 1, l2, prefix = 'Progress:', suffix = 'Complete', length = 50)  
        except Exception as e:
            print(e)
            pass




# This will take an inordinate amount of time - 277524 images, will be easier to use a cloud computer (global accessible neural net - possible idea)
# New Idea: We can take the first 1000 images and use them for training, the latter option would take very long
create_training_data()

print(trainingData)
#print("\n" + len(trainingData))

import random

random.shuffle(trainingData)

for sample in trainingData:
    print(sample[1])
 

X = []
y = []

for features, label in trainingData:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 50, 50, 1)

import pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# image_name = imagePatches[0]


# 
# 
# print(classZero)
# (train_images, train_labels) = imagePatches 

# # Normalize pixel values to be between 0 and 1
# train_images = imagePatches / 255.0
# print(train_images)

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
# plt.show()


# def plotImage(image_location):
#     image = cv2.imread(image_name)
#     image = cv2.resize(image, (50, 50))
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()
    
# plotImage(image_name)




