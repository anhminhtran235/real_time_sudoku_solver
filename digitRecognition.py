# Disclaimer: In this .py file, I use Keras to build a Convolutional Neural Network
# with the exact same architecture as the one proposed by F.Chollet as an example 
# in his Keras library for the Hand-written Digit Recognition problem:
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
#
# I added some preprocessing steps, which I learned from these 2 sources:
# 1/ https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
# 2/ https://www.youtube.com/watch?v=j-3vuBynnOE&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=2


from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import numpy
import os
import random
import cv2
from scipy import ndimage

# get_best_shift and shift is used to centralize an image according to its center of mass
def get_best_shift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = numpy.round(cols/2.0-cx).astype(int)
    shifty = numpy.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

# get_best_shift and shift is used to centralize an image according to its center of mass
def shift(img,sx,sy):
    rows,cols = img.shape
    M = numpy.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

# Centralize an image according to its center of mass
# We will do the same thing with Sudoku Digit 
# so that every single digit image is in the SAME form
# Without this step, the accuracy drops from nearly 100% to 70-80%
def shift_according_to_center_of_mass(img):
    img = cv2.bitwise_not(img)

    # Centralize the image according to center of mass
    shiftx,shifty = get_best_shift(img)
    shifted = shift(img,shiftx,shifty)
    img = shifted

    img = cv2.bitwise_not(img)
    return img

batch_size = 128
num_classes = 9
epochs = 35

# input image dimensions
img_rows, img_cols = 28, 28

DATADIR = "DigitImages"
CATEGORIES = ["1","2","3","4","5","6","7","8","9"]

# Read training Data (I stored it in folders named 1, 2,..., 9 in the same directory)
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_rows, img_cols))
            new_array = shift_according_to_center_of_mass(new_array)
            training_data.append([new_array, class_num])

create_training_data()

# Mix data up
random.shuffle(training_data)

# Split 80-20
x_train = []
y_train = []
x_test = []
y_test = []
for i in range(len(training_data)*8//10):
    x_train.append(training_data[i][0])
    y_train.append(training_data[i][1])
for i in range(len(training_data)*8//10,len(training_data)):
    x_test.append(training_data[i][0])
    y_test.append(training_data[i][1])

# Reshape
x_train = numpy.array(x_train)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = numpy.array(x_test)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the data
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save_weights('digitRecognition.h5')


