import cv2
import sys
import numpy as np

# Silence tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup():
    global type_of_distraction_dict, cap

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each type of distraction a label
    type_of_distraction_dict = {0: "Other activities", 1: "Safe driving", 2: "Talking on the phone", 3: "Texting on the phone", 4: "Turning"}
    
    # start the webcam feed
    cap = cv2.VideoCapture(0)

class NoFaceDetectedException(Exception):
    def msg():
        return "No face detected on the screen"

def getDistractionType():
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        return 0
    
    facecasc = cv2.CascadeClassifier(sys.path[0] + '/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    prediction = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)

    data = {}
    if type(prediction) is np.ndarray:
        for distraction_type in type_of_distraction_dict:
            data[type_of_distraction_dict[distraction_type]] = round(prediction[0][distraction_type].item(), 5)
    else:
        raise NoFaceDetectedException

    return data