import mahotas as mt
import cv2
import glob
import numpy as np
from sklearn.naive_bayes import GaussianNB
from os import listdir
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def extract_features(img):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(img)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean


# load the dataset
train_path = 'BRACOL_dataset'
train_names = listdir(train_path)

# empty list to hold feature vectors and train labels
train_features = list()
train_labels = list()

# loop over the training dataset
print("[STATUS] Started extracting haralick textures...")
for train_name in train_names:
    cur_path = train_path + "/" + train_name
    cur_label = train_name
    i = 1

    for file in glob.glob(cur_path + "/*.jpg"):
        print(f"Processing Image - {i} in {cur_label}")
        # read the training image
        image = cv2.imread(file)

        # extract haralick texture from the image
        features = extract_features(image)

        # append the feature vector and label
        train_features.append(features)
        train_labels.append(cur_label)

        # show loop update
        i += 1

# create the classifier
print("[STATUS] Creating the classifier...")
gnb = GaussianNB()

# fit the training data and labels
print("[STATUS] Fitting data/label to model...")
gnb.fit(train_features, train_labels)

test_path = 'Testes'
test_names = listdir(test_path)

print("[STATUS] Started making predictions...")
preds = list()
test_labels = list()
for test_name in test_names:
    cur_path = test_path + "/" + test_name
    cur_label = test_name
    i = 1

    for file in glob.glob(cur_path + "/*.jpg"):
        test_labels.append(test_name)
        print(f"Predicting Image - {i} in {cur_label}")

        # read the training image
        image = cv2.imread(file)

        # extract haralick texture from the image
        features = extract_features(image)

        # evaluate the model and predict label
        prediction = gnb.predict(features.reshape(1, -1))[0]
        preds.append(prediction)

        i += 1

# Calculate accuracy, precision, recall and f1 scores
accuracy = accuracy_score(test_labels, preds)
precision = precision_score(test_labels, preds, average="micro")
recall = recall_score(test_labels, preds, average="micro")
f1 = f1_score(test_labels, preds, average='micro')

print(f'Accuracy Score: {accuracy:.4f}')
print(f'Precision Score: {precision:.4f}')
print(f'Recall Score: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Save the results in a file
file = open('Accuracy Scores - Haralick.txt', 'a')
file.write('Haralick Features - BRACOL\n')
file.write(f'Accuracy: {accuracy:.4f}\n')
file.write(f'Precision: {precision:.4f}\n')
file.write(f'Recall: {recall:.4f}\n')
file.write(f'F1: {f1:.4f}\n\n')
file.close()
