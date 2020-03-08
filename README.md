# Face-Recognition
Facial Recognition System using Dlib and Openface

A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. There are multiple methods in which facial recognition systems work, but in general, they work by comparing selected facial features from given image with faces within a database.

Face Detection is done using Dlib library.Dlib for face detection uses a combination of HOG (Histogram of Oriented Gradient) & Support Vector Machine (SVM) which is trained on positive and negative images (meaning there are images that have faces and ones that don’t).

After we isolate the image from the background and preprocess it using dlib we need to find a way to represent the face in numerical embedding. We can represent it using pretrained deep neural network OpenFace which produce 128 facial embeddings that represent a generic face.

After getting the embedding of each of the images in dataset , we can train a machine learning model or NN model as a classifier taking face embedding as train data and names as class label of train data. 
