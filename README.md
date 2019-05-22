# Real_Time_Ghost_Imaging
Attempting to create a code which will run in situ with a ghost imaging experiment to identify an object as soon as possible, without it fully being formed.

There are two folders with images:

(1) The "Training Data.zip" folder currently holds 2000 images, 1000 variations of three and 1000 variations of four. The two folders are called in the "number_recognition_data_input.py" code.

(2) The "Predict against.zip" folder contains falsely created images of 3's and 4's that have varying levels of noise on the image. This is used in the "number_recognition_num_predict.py" code.

And currently there are three program files included:

(1)The first file is the "number_recognition_data_input.py" which turns the training data (3's and 4's) into arrays and pickles it into an X and y pickle. I have 15000 of each 3 and 4 where the actual number is shifted, rotated and dialated so there should be no issue with the data being the same that I see. 

(2)The second file, "number_recognition_cnn.py" is the cnn that seems to overfit the data. This is where the major issue lies I believe, No combination of cnn seems to not produce an accuracy of over 90% within three epochs.

(3)The "number_recognition_num_predict.py" file is to purely test against falsely created data to see if the cnn can accurately identify a 3 from a 4.

This is the journal article I have found that might be able to help:
https://www.nature.com/articles/s41598-017-18171-7
