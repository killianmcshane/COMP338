import numpy as np
import cv2 as cv
import glob
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

image_list = []
all_des = []
print("GRABBING IMAGES")
for filename in glob.glob(r'C:\\Users\\Killian\\Documents\\Uni\\Third Year\\COMP338\\Assignment 1\\COMP338_Assignment1_Dataset\\Training\\dog\\*.jpg'
): #gets all images
    im = cv.imread(filename)
    image_list.append(im)

print("GETTING DESCRIPTORS")
for img in image_list: # Gets keypoints and descriptors for image (to be replaced with tylers code)
    grey= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kp, des = cv.SIFT.create().detectAndCompute(grey, None) 
    all_des.append(des)

all_desarray=np.array(all_des, dtype=np.ndarray) #changes descriptor list to array
all_des=np.concatenate(all_des, axis=0)
print('TRAINING KMEANS DOGS, WILL TAKE A FEW MINS...')
km = KMeans(n_clusters=100, random_state=0).fit(all_des)


############################################################################################
###########################     STEP 4 & 5 BELOW  ##########################################
############################################################################################
    # note: needs a fully trained kmeans clustering model (km) to be generated before the code
    #       below can run (it's used in the histogram creation).

'''
    Classes are converted to numbers to reduce computation cost.
        airplane = 0
        cars = 1
        dog = 2
        faces = 3
        keyboard = 4
'''

'''
    Reads in all training images, and labels each image according to its class.
'''
def readTrainingImages():
    root_dir = 'C:\\Users\\Killian\\Documents\\Uni\\Third Year\\COMP338\\Assignment 1\\COMP338_Assignment1_Dataset\\Training\\'
    folders = glob.glob(root_dir + '*')

    training_images = []
    training_labels = []

    for folder in folders:
        for image in glob.glob(folder+'/*.jpg'):
            training_images.append(cv.imread(image))
            if 'airplane' in folder:
                training_labels.append(0)
            elif 'car' in folder:
                training_labels.append(1)
            elif 'dog' in folder:
                training_labels.append(2)
            elif 'face' in folder:
                training_labels.append(3)
            elif 'key' in folder:
                training_labels.append(4)

    return training_images, training_labels


'''
    Reads in all testing images, and labels each image according to its class to determine accuracy of predictions.
'''
def readTestingImages():
    root_dir = 'C:\\Users\\Killian\\Documents\\Uni\\Third Year\\COMP338\\Assignment 1\\COMP338_Assignment1_Dataset\\Test\\'
    folders = glob.glob(root_dir + '*')

    testing_images = []
    testing_labels = []
    for folder in folders:
        for image in glob.glob(folder+'/*.jpg'):
            testing_images.append(cv.imread(image))
            if 'airplane' in folder:
                testing_labels.append(0)
            elif 'car' in folder:
                testing_labels.append(1)
            elif 'dog' in folder:
                testing_labels.append(2)
            elif 'face' in folder:
                testing_labels.append(3)
            elif 'key' in folder:
                testing_labels.append(4)
    
    return testing_images, testing_labels


'''
    Converts all training images to normalised histograms.
'''
def getTrainingHistograms(training_images, training_labels, km):
    X_train = []
    y_train = training_labels

    for image in training_images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #Convert to grayscale 
        kp, descript = cv.SIFT.create().detectAndCompute(gray, None) # Gets keypoints and descriptors for image (to be replaced with tylers code)
        histogram, bin_edges=np.histogram(km.predict(descript)) #Histogram as feature vector

        X_train.append(histogram)

    return X_train, y_train


'''
    Converts all testing images to normalised histograms.
'''
def getTestingHistograms(testing_images, testing_labels):
    X_test = []
    y_test = testing_labels
    for image in testing_images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Convert to grayscale 
        kp, descript = cv.SIFT.create().detectAndCompute(gray, None) # Gets keypoints and descriptors for image (to be replaced with tylers code)
        test_histogram, bin_edges=np.histogram(km.predict(descript))
        
        X_test.append(test_histogram)

    return X_test, y_test


'''
    Classifies every histogram from the testing image set with a class prediction.
    Prints the confusion matrix, error rate and classification errors for each class.
'''
def getClassifications(X_train, y_train, X_test, y_test):  
    k_nn = KNeighborsClassifier(n_neighbors=3)
    k_nn.fit(X_train, y_train)
    predictions = k_nn.predict(X_test)
    accuracy = k_nn.score(X_test, y_test)
    classes = ["airplanes", "cars", "dog", "faces", "keyboard"]
    
    print("\nERROR RATE: " + str(1-accuracy))
    print(confusion_matrix(y_test, predictions))

    # Splits the labels & predictions into their seperate classes to evaluate.
    y_test = np.asarray(np.array_split(y_test, 5))
    predictions = np.asarray(np.array_split(predictions, 5))
    print("\n")
    error_total = 0
    class_type = 0

    for sublist_y, sublist_p in zip(y_test, predictions):
        error_count = 0
        
        for i in range(0, len(sublist_y)):
            if sublist_p[i] != sublist_y[i]:
                error_count += 1
                
        error_total += error_count
        print("Class Label [" + classes[class_type] + "] Classification Errors: " + str(error_count))
        class_type += 1
    print("\nTotal Classification Errors: " + str(error_total))


# Reads images into arrays.
training_images, training_labels = readTrainingImages()
testing_images, testing_labels = readTestingImages()

# Creates histograms.
X_train, y_train = getTrainingHistograms(training_images, training_labels, km)
X_test, y_test = getTestingHistograms(testing_images, testing_labels)

# Classifies the histograms.
getClassifications(X_train, y_train, X_test, y_test)

