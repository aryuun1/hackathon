import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
#complete eye and face detect.
while 1:
    ret ,img = cap.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h,x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes: 
          cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

    cv2.imshow('img',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break


faces  = detector(gray)
for face in faces:
   

  landmarks = predictor(gray, face)

  left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41],landmarks)
  right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47],landmarks)
  blinking_ratio = (left_eye_ratio+right_eye_ratio)/2

  if blinking_ratio>5.7:
     cv2.putText(frame,"Blinking",(50, 150),font, 7,(255,0,0))

  
  left_eye_region = np.array([(landmarks.part(36).x,landmarks.part(36).y),
                              (landmarks.part(37).x,landmarks.part(37).y),
                              (landmarks.part(38).x,landmarks.part(38).y),
                              (landmarks.part(39).x,landmarks.part(39).y),
                              (landmarks.part(40).x,landmarks.part(40).y),
                              (landmarks.part(41).x,landmarks.part(41).y)],np.int32)

  print(left_eye_region)



cap.release()
cv2.destroyAllWindows()
      

