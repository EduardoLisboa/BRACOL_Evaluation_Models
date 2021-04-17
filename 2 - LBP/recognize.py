# import the necessary packages
from LBP import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imutils import paths
import time
import cv2
import os

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

start = time.time()
print('[STATUS] Getting LBP vectors to training images')
i = 1
for imagePath in paths.list_images('BRACOL_dataset'):
    # load the image, convert it to grayscale, and describe it
    print(f'Processing image {i} - {imagePath}')
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)
    i += 1

print('[STATUS] DONE')
print('[STATUS] Fitting model')
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)
print('[STATUS] DONE')

print('[STATUS] Making predictions for the test images')
preds = list()
val = list()
# loop over the testing images
i = 1
for imagePath in paths.list_images('Testes'):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    print(f'Predicting image {i} - {imagePath}')
    folder = imagePath.split('\\')
    val.append(folder[1])
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))

    preds.append(prediction[0])
    i += 1

print('[STATUS] DONE')
print(f'Elapsed time: {time.time() - start:.2f}')

print(preds)

# Calculate accuracy, precision, recall and f1 scores
accuracy = accuracy_score(val, preds)
precision = precision_score(val, preds, average="micro")
recall = recall_score(val, preds, average="micro")
f1 = f1_score(val, preds, average='micro')

print(f'Accuracy Score: {accuracy:.4f}')
print(f'Precision Score: {precision:.4f}')
print(f'Recall Score: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Save the results in a file
file = open('Accuracy Scores - LBP.txt', 'a')
file.write('Local Binary Patterns - BRACOL\n')
file.write(f'Accuracy: {accuracy:.4f}\n')
file.write(f'Precision: {precision:.4f}\n')
file.write(f'Recall: {recall:.4f}\n')
file.write(f'F1: {f1:.4f}\n\n')
file.close()
