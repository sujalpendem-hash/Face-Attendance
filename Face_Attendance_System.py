# Face Attendance System using OpenCV
# Install required libraries before running:
# pip install opencv-python face_recognition numpy pandas

import cv2
import face_recognition
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Folder containing images
path = 'ImagesAttendance'

images = []
classNames = []

# Read images from folder
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# Function to encode faces
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

# Function to mark attendance
def markAttendance(name):
    file_name = 'Attendance.csv'

    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write('Name,Time\n')

    with open(file_name, 'r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')

            f.writelines(f'\n{name},{dtString}')
            print(f'Attendance Marked for {name}')

# Encode known faces
print("Encoding Started...")
encodeListKnown = findEncodings(images)
print("Encoding Complete")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # Resize image for faster processing
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # Detect faces
    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc

            # Scale back up
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)

            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255, 255, 255), 2)

            markAttendance(name)

    cv2.imshow('Face Attendance System', img)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()