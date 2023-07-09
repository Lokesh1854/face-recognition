import os
import cv2 as cv
import numpy as np

DIR = r'Photos_Train'                           #   to extract the names of person
DIR_1 = r'Photos_Check'                         #   folder of images, model to be tested on

people = []                                     #   will store the name of 

for i in os.listdir(DIR):                       #   getting the name of persons index wise, as labels list has index only
    people.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')    #   haar_cascade to detect face from image to be tested on

features = np.load('features.npy', allow_pickle=True)   #   loading the previously created features list
labels = np.load('labels.npy')                          #   loading the previously created labels list

face_recognizer = cv.face.LBPHFaceRecognizer_create()   #   opencv built in face recognizer
face_recognizer.read('face_trained.yml')                #   loading the trained model

for j in os.listdir(DIR_1):                             #   testing the model on all the images present in directory
    img_path = os.path.join(DIR_1,j)                    #   joining the path to image
    img = cv.imread(img_path)                           #   reading the image from img_path
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)          #   converting it into grayscale

    face_rect = haar_cascade.detectMultiScale(gray,1.05,4)  # detecting the face

    for(x,y,w,h) in face_rect:                          #   extracting the face
        face_roi = gray[y:y+h, x:x+h]           
        label, confidence = face_recognizer.predict(face_roi)       #   extraced image sent for recognition, it gives person_name with confidence percentage
        print(people[label],confidence)                             #   display the person with confidence
        cv.putText(img,str(people[label]),(x,y),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=0)  #   person detected is also marked on image, for ease of viewing
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)     #   rectangular box at detected image

    cv.imshow(j,img)    #   visualising the image

cv.waitKey(0)