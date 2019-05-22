#Inputs data and pickles the x and y as X.pickle and y.pickle respectively

import numpy as np
import matplotlib.pyplot as plt
import keras.preprocessing.image
import matplotlib.cm as cm  
import os
import cv2
import random
import pickle

DATADIR = "Numbers"

CATEGORIES = ["Hold3", "Hold4"]

for category in CATEGORIES:  # do 3 and 4
    path = os.path.join(DATADIR,category)  # create path to 3 and 4
    for img in os.listdir(path):  # iterate over each image per 3 and 4
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!

IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:  # do three and fours

        path = os.path.join(DATADIR,category)  # create path to 3 and 4
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=3 1=4

        for img in (os.listdir(path)):  # iterate over each image per three and four
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

## generate new images via rotations, translations, zoom using keras. Only turn on to add more to training data
#def generate_images(imgs):
#    
#    # rotations, translations, zoom
#    image_generator = keras.preprocessing.image.ImageDataGenerator(
#        rotation_range = 20, width_shift_range = 0.2 , height_shift_range = 0.2)
#
#    # get transformed images
#    imgs = image_generator.flow(imgs.copy(), np.zeros(len(imgs)),
#                                batch_size=len(imgs), shuffle = False).next()    
#  
#    return imgs[0]    
#
## create randomized images
#for i in range(5000):
#    n = np.random.randint(0, X.shape[0] - 2)
#    plt.figure(figsize=(9.6, 9.6), dpi=100)
#    plt.axis('off')
#    plt.imshow(generate_images(X[n:n + 1]).reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
#    plt.savefig('Save 4/' + str(i) + '.png', facecolor='k', dpi=100)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()