import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping

batch_size = 16
epochs = 50

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="Train", target_size=(224, 224), batch_size=batch_size)
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="Test", target_size=(224, 224), batch_size=batch_size)

model = Sequential()
model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=2, activation="softmax"))

print(model.summary())

opt = Adam(lr=0.001)
history = model.compile(
    optimizer=opt,
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='accuracy', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
start = time.time()
hist = model.fit_generator(generator=traindata, validation_data=testdata,
                           validation_steps=10, epochs=epochs, callbacks=[checkpoint, early])
print(f'\nElapsed time: {time.time() - start:.2f} seconds')

model.save(f'Models/Batch Size {batch_size} - Epochs {epochs} - My_VGG16.h5')
