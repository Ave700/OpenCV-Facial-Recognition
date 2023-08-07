import cv2

##use case 1 ------------ easy facial recognition ---------
#import image
image = cv2.imread('.\images\people1.jpg')
#resize and recolor
image = cv2.resize(image, (800, 600))
mono_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



#time to detect faces
face_detector = cv2.CascadeClassifier('.\cascades\haarcascade_frontalface_default.xml')
# Without scale factor we get 1 false positive.
# With it set to 1.09 we get no false positives
detections = face_detector.detectMultiScale(mono_image, scaleFactor = 1.09)


#draw bounding boxes for faces
for (x_pos, y_pos, width, height) in detections:
    cv2.rectangle(image, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 0, 255), 2)


cv2.imshow('test 1 easy',image)
#wait until the user presses a key
cv2.waitKey(0)


##use case 2 ------------- harder image set -------------
image = cv2.imread('.\images\people2.jpg')
mono_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# With min neighbors and this scale factor we get no false positives
# but we fail to detect all faces
detections = face_detector.detectMultiScale(mono_image, scaleFactor=1.2, minNeighbors=7)

for (x_pos, y_pos, width, height) in detections:
    cv2.rectangle(image, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 0, 255), 2)


cv2.imshow('test 2 harder',image)
cv2.waitKey(0)


#cleanup windows
cv2.destroyAllWindows()