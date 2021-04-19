import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime



path='imgs'
images=[]
classNames=[]
#images in the img folder, making a list
list=os.listdir(path)
print(list)
for cls in list:
    curImg=cv.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def encodings(images):
    encodList=[]
    for img in images:
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodList.append(encode)
    return encodList


def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        dataList=f.readlines()
        nameList=[]
        for line in dataList:
            enter=line.split(',')
            nameList.append(enter[0])

        if name not in nameList:
            now=datetime.now()
            tstr=now.strftime('%H:%M:%S')
            dstr=now.strftime('%m/%d/%Y')
            f.writelines(f'\n{name},{dstr},{tstr}')


encodeListKnown=encodings(images)
print('Encoding Complete')

#initilaising Webcam
cap=cv.VideoCapture(0,cv.CAP_DSHOW)

while cap.isOpened():
    success,img=cap.read()
    #resizing in order to speed up our processes
    imgS=cv.resize(img,(0,0),None,0.25,0.25)
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)
    #since there might be multiple faces, and we take frames for all of them
    facesCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)
    #print(len(facesCurFrame))
    #print(len(encodeCurFrame))
    #cv.imshow("Vidhi",imgS)
    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(matches,faceDist)
        matchIndex=np.argmin(faceDist)

        if matches[matchIndex]:
            name=classNames[matchIndex]
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),2,cv.FILLED)
            cv.putText(img,name,(x1+6,y2-6),cv.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
            markAttendance(name)
    #cv.imshow('Webcam',img)

