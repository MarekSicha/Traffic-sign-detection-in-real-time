# Author: Marek Sicha

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =============================================================================
# Setting and importing traininig parameters, data
# =============================================================================
path = 'path to your dataset'  # folder path with all images split in folders
bach_size_val = 64
epochs = 30
imageDimension = (32, 32, 3)
test_ratio = 0.1      # 10% will split for testing => 90%  remain for training
valid_ratio = 0.2     # from 90% for traininig  will be 20% for validation

# =============================================================================

# =============================================================================
# Processing with images
# =============================================================================
images = []
classNo = []
mylist = os.listdir(path)
n_of_classes = len(mylist)
print('Total classes detected', n_of_classes)
print('Importing classes.....')
for directory in os.listdir(path):
    myfilelist = os.listdir(path+'/'+directory)
    for img in myfilelist:
        if img[-4:] == '.ppm':
            curIMG = cv2.imread(path+'/'+directory+"/"+img)
            curIMG = cv2.resize(curIMG, (32, 32))
            curIMG = cv2.cvtColor(curIMG, cv2.COLOR_BGR2GRAY)
            curIMG = cv2.equalizeHist(curIMG)
            curIMG = curIMG.reshape(curIMG.shape[0], curIMG.shape[1], 1)
            images.append(curIMG)
            classNo.append(str(int(directory)))
        print(str(int(directory)), sep=' ', end=' ', flush=True)
print(' ')
images = np.array(images)
classNo = np.array(classNo)
# =============================================================================

# =============================================================================
# Spliting data
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=test_ratio)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_ratio)
# =============================================================================

# =============================================================================

# =============================================================================
print('Data shapes')
print('Train', sep=' ', end=' ', flush=True)
print(X_train.shape, y_train.shape)
print('Validation', sep=' ', end=' ', flush=True)
print(X_valid.shape, y_valid.shape)
print('Test', sep=' ', end=' ', flush=True)
print(X_test.shape, y_test.shape)

# =============================================================================

# =============================================================================
y_train = to_categorical(y_train, n_of_classes)
y_test = to_categorical(y_test, n_of_classes)
y_valid = to_categorical(y_valid, n_of_classes)


# =============================================================================
# CNN Model
# =============================================================================
def myModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(5, 5), activation='relu',
                                     input_shape=(imageDimension[0], imageDimension[1], 1)))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(n_of_classes, activation='softmax'))  # Output layer has output == n_of_classes

    # Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# =============================================================================
# Train
# =============================================================================
model = myModel()
print(model.summary())
history = model.fit(X_train, y_train, batch_size=bach_size_val,
                    epochs=epochs, validation_data=(X_test, y_test))

# =============================================================================

# =============================================================================
# Ploting the training graphs
# =============================================================================
plt.figure(0)
plt.plot(history.history['accuracy'], label='training')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.grid()
plt.savefig('Accuracy.svg')
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()
plt.savefig('Loss.svg')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score', score[0])
print('Test Accuracy', score[1])

# =============================================================================


# =============================================================================
# Saving trained model
# =============================================================================
tf.saved_model.save(model, 'saved_model/')
