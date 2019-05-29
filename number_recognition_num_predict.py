#Takes the Neural Net created by program "number_recognition.py" to predict a given photo

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

classifier = load_model('4cnn.h5')

test_image = image.load_img('corrupted/test4.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)[0] 

if result == 0:
    prediction = '4'
else:
    prediction = 'Not 4'
    
result = tuple(result)[0]
print ("prediction: {}".format(prediction) )

















