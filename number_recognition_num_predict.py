#Takes the Neural Net created by program "number_recognition.py" to predict a given photo

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ["3", "4"]

def prepare (filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("4_or_not.model")

prediction = model.predict([prepare('corrupted/SmallNumber3.png')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/SmallNumber4.png')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/3-64-0.tif')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/4-32-5.tif')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/3-64-10.tif')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/4-32-15.tif')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/3-64-20.tif')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/4-32-25.tif')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/3-64-30.tif')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/4-32-35.tif')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/3-64-40.tif')])
print(CATEGORIES[int(prediction[0][0])])

prediction = model.predict([prepare('corrupted/4-32-45.tif')])
print(CATEGORIES[int(prediction[0][0])])