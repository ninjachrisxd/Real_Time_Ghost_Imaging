# Real_Time_Ghost_Imaging
Attempting to create a code which will run in situ with a ghost imaging experiment to identify an object as soon as possible, without it fully being formed.

Currently there are three program files included:

The first file is the "number_recognition_data_input.py" which turns the training data (3's and 4's) into arrays and pickles it into an X and y pickle. I have 15000 of each 3 and 4 where the actual number is shifted, rotated and dialated so there should be no issue with the data being the same that I see. 

The second file, "number_recognition_cnn.py" is the cnn that seems to overfit the data. This is where the major issue lies I believe, No combination of cnn seems to not produce an accuracy of over 90% within three epochs.

The "number_recognition_num_predict.py" file is to purely test against falsely created data to see if the cnn can accurately identify a 3 from a 4

