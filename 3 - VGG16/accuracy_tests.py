from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os

saved_model = load_model("vgg16_1.h5")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Read the data from the csv
new_data = pd.read_csv('dataset.csv')

# Store the id and stress values
data_id = new_data['id']
data_stress = new_data['predominant_stress']
data_id_list = data_id.tolist()
data_stress_list = data_stress.tolist()

# Cicle through all the test files to make a list of all the
# correct values for the validation
validation = list()
for file in os.listdir(os.path.join('Testes_ok', '')):
    split = file.split('.')
    file_id = split[0]

    if int(file_id) in data_id_list:
        id_index = data_id_list.index(int(file_id))
        validation.append(0 if data_stress_list[id_index - 1] == 0 else 1)

# Make all the predictions and save in a list
predictions = list()
for file in os.listdir('Testes_ok'):
    img = image.load_img(f'Testes_ok/{file}', target_size=(224, 224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)

    output = saved_model.predict(img)
    if output[0][0] > output[0][1]:
        predictions.append(1)
    else:
        predictions.append(0)

# Calculate accuracy, precision, recall and f1 scores
accuracy = accuracy_score(validation, predictions)
precision = precision_score(validation, predictions, average='micro')
recall = recall_score(validation, predictions, average='micro')
f1 = f1_score(validation, predictions, average='micro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')

# Save the results in a file
file = open('Accuracy Scores - VGG16.txt', 'a')
file.write('VGG16 retrained model\n')
file.write('Batch Size - 16 | Epochs - 50\n')
file.write(f'Accuracy: {accuracy:.4f}\n')
file.write(f'Precision: {precision:.4f}\n')
file.write(f'Recall: {recall:.4f}\n')
file.write(f'F1: {f1:.4f}\n\n')
file.close()
