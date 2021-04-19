# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #print(f'Hi, {name}')
    #print(name)  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#importing libraraies


import cv2 as cv
import face_recognition
import numpy as np

#training image
imgs1=face_recognition.load_image_file('imgs/v1.jpg')
imgs1=cv.cvtColor(imgs1,cv.COLOR_BGR2RGB)
#testing image
imgt=face_recognition.load_image_file('imgtest/v2.jpg')
imgt=cv.cvtColor(imgt,cv.COLOR_BGR2RGB)

#finding faces in inmages and finding their encodings
faceloc=face_recognition.face_locations(imgs1)[0]
encods=face_recognition.face_encodings(imgs1)[0]
cv.rectangle(imgs1,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
faceloct=face_recognition.face_locations(imgt)[0]
encodt=face_recognition.face_encodings(imgt)[0]
cv.rectangle(imgt,(faceloct[3],faceloct[0]),(faceloct[1],faceloct[2]),(255,0,255),2)

result=face_recognition.compare_faces([encods],encodt)
#finding how similar the images are(lower the distance, better the match)
facedist=face_recognition.face_distance([encods],encodt)
print(result,facedist) #output true as both faces match
#creating text on a picture
cv.putText(imgt,f'{result} {round(facedist[0],2)}',(25,25),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)

cv.imshow('Sundar Train',imgs1)
cv.imshow('Sundar Test',imgt)
cv.waitKey(0)

