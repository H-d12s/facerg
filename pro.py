import cv2
import numpy as np
import face_recognition
imgelon = face_recognition.load_image_file("elon_musk_royal_society.jpg")
cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgelont= face_recognition.load_image_file("the-daily-schedule-of-elon-musk-from-a-morning-donut-to-upping-his-sleep-hours.webp")
imgelonr=cv2.resize(imgelon,(300,300))
#[0] because we are passing onlyone image and sendong the first image index

facloc= face_recognition.face_locations(imgelonr)[0]
facencode=face_recognition.face_encodings(imgelonr)[0]

facloct= face_recognition.face_locations(imgelont)[0]
facencodet=face_recognition.face_encodings(imgelont)[0]

cv2.rectangle(imgelonr,(facloc[3],facloc[0]),(facloc[1],facloc[2]),(255,0,255),2)
cv2.rectangle(imgelont,(facloct[3],facloct[0]),(facloct[1],facloct[2]),(255,0,255),2)

cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
result=face_recognition.compare_faces([facencode],facencodet)
facedis=face_recognition.face_distance([facencode],facencodet)
print(result,facedis)
cv2.putText(imgelonr,f'{result},{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
cv2.imshow("elon",imgelonr)
cv2.imshow("elont",imgelont)
cv2.waitKey(0)