import cv2

#------ image 1 face and eye detector----
image = cv2.imread('.\images\people1.jpg')

#resize and recolor
image = cv2.resize(image, (1333, 1000))
mono_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



#stup detectors
face_detector = cv2.CascadeClassifier('.\cascades\haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('.\cascades\haarcascade_eye.xml')

## A note on parameters: Tuning parameters sucks. Tuning 3 parameters for each type of
# detection for each image and resolution is super inefficient

#draw bounding boxes for faces
face_detections = face_detector.detectMultiScale(mono_image, scaleFactor = 1.25, minSize = (40,40))
for (x_pos, y_pos, width, height) in face_detections:
    cv2.rectangle(image, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 0, 255), 2)

#draw bounding boxes for eyes
eye_detections = eye_detector.detectMultiScale(mono_image, scaleFactor = 1.1 , minNeighbors=7, maxSize=(50,50))
for (x_pos, y_pos, width, height) in eye_detections:
    cv2.rectangle(image, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 255, 255), 2)

cv2.imshow('Face and Eye Recognition',image)
#wait until the user presses a key
cv2.waitKey(0)