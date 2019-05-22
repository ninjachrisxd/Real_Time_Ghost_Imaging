#Takes the data from the pickles (created in program "number_recognition_data_input") and puts it into the Neural Net to train

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
import pickle
import time

NAME = "Test_CNN"

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0
num_classes = 1

model = Sequential()

model.add(Conv2D(16, (2, 2), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (20, 20)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#model.add(Conv2D(64, kernel_size=(20, 20),
#                 activation='relu',
#                 input_shape=X.shape[1:]))
#model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(Conv2D(32, (20, 20), activation='relu'))
##model.add(MaxPooling2D(pool_size=(4, 4)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(64, activation='relu'))
##model.add(Dropout(0.5))
##model.add(Dense(num_classes, activation='sigmoid'))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit(X, y,
          batch_size=32,
          epochs=3,
          validation_split=0.3,
          callbacks=[tensorboard])

model.save('4_or_not.model')