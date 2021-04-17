from time import time
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# For the accuracy tests
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Generate the dataset
image_size = (512, 512)
batch_size = 32
epochs = 50

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "BRACOL_dataset",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "BRACOL_dataset",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

num_classes = 2

# Using data augmentation and dropout to enhance results
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(512, 512, 3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

# Create the model
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compile the created model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit the model and measure training time
start = time()
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
print(f'\nElapsed time: {time()-start:.2f}\n')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Plot a graphic for training accuracy and loss and validation accuracy and loss
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Save the plot
plt.savefig(f'Plots/Plot - Batch Size {batch_size} - Epochs {epochs}.png')

plt.show()

# Save the model
model.save(f'Models/Batch Size {batch_size} - Epochs {epochs}.h5')


"""
-----------------------------------------------------------------------------------------
------------------------------------  ACCURACY TESTS ------------------------------------
-----------------------------------------------------------------------------------------
"""

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
for file in os.listdir(os.path.join('Testes', '')):
    split = file.split('.')
    file_id = split[0]
    # print(file_id)

    if int(file_id) in data_id_list:
        id_index = data_id_list.index(int(file_id))
        validation.append(0 if data_stress_list[id_index] == 0 else 1)

# Make the predictions and store them in a list
predictions = list()
for file in os.listdir(os.path.join('Testes', '')):
    img_path = os.path.join('Testes', file)
    img = keras.preprocessing.image.load_img(img_path, target_size=(512, 512))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0])
    predictions.append(0 if class_names[np.argmax(score)] == 'healthy' else 1)

# Calculate accuracy, precision, recall and f1 scores
accuracy = accuracy_score(validation, predictions)
precision = precision_score(validation, predictions, average='binary')
recall = recall_score(validation, predictions, average='binary')
f1 = f1_score(validation, predictions, average='binary')

print(f'Batch Size - {batch_size} | Epochs - {epochs}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}\n')

# Save the results in a file
file = open('Accuracy Scores.txt', 'a')
file.write(f'Batch Size - {batch_size} | Epochs - {epochs}\n')
file.write(f'Accuracy: {accuracy:.4f}\n')
file.write(f'Precision: {precision:.4f}\n')
file.write(f'Recall: {recall:.4f}\n')
file.write(f'F1: {f1:.4f}\n\n')
file.close()
