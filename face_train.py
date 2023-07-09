import os
import cv2 as cv
import numpy as np

#   we will use openCV built in face recognizer

#   copied tha path of folder Photos, a list people is created which will store names of other files inside it

DIR = r'Photos_Train'
people = []                 #   will store name of folders(people) inside Photo_Train to train the model

for i in os.listdir(DIR):   #   will traverse to directory and add it in people[]
    people.append(i)

#   people[] contains name of all the folders inside

features =[]    #   will store face extracted using face detection
labels = []     #   will store corresponding label (person) whose face it is

def create_train(features,labels):

    for person in people:                                       #   iterated in people folder to get the name of person
        path = os.path.join(DIR,person)                         #   joins the directory, to open folder of that person
        label = people.index(person)                            #   label stores the index

        for img in os.listdir(path):                            #   now folder of specific person is iterated
            
            img_path = os.path.join(path,img)                   #   path joined to get the path of each image
            img_array = cv.imread(img_path)                     #   reading image from img_path
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)     #   conerting it to grayscale

            haar_cascade = cv.CascadeClassifier('haar_face.xml')    #   using haar_cascade to extract image
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 3)   #   face detected

            for (x,y,w,h) in faces_rect:                        #   detected face is taken out 
                face_roi = gray[y:y+h, x:x+h]                   #   using coordinates stored in face_rect

                features.append(face_roi)                       #   extracted face is stored in features list
                labels.append(label)                            #   corresponding label (person name) is stored in labels list

create_train(features,labels)                                   #   calling the function created to 

features = np.array(features,dtype='object')                    #   features and labels list are
labels = np.array(labels)                                       #   converted into numpy array

face_recognizer = cv.face.LBPHFaceRecognizer_create()           #   calling openCV built-in face recognizer 
face_recognizer.train(features,labels)                          #   training the model

face_recognizer.save('face_trained.yml')                        #   saving the file for future use

np.save('features.npy', features)                               #   also saving the numpy arrays for future use
np.save('labels.npy', labels)

print('training done')                                          #   will print training done, when model is successfully trained