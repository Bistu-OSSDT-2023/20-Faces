import cv2 as cv
import numpy as np


def face_detect_demo(img):
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Load the face cascade classifier
    face_detector = cv.CascadeClassifier('D:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale image
    faces = face_detector.detectMultiScale(gray, 1.25)
    face_count = len(faces)  # Count the number of faces detected

    # Load the replacement image
    replace_img = cv.imread('5.png')

    for x, y, w, h in faces:
        # Replace the face region with the replacement image
        img[y:y + h, x:x + w] = cv.resize(replace_img, (w, h))

    # Draw rectangles and display the number of faces
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)

    cv.putText(img, f"Faces: {face_count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow('result', img)


# Connect to the webcam
cap = cv.VideoCapture(0)
cv.namedWindow('result')  # Create a window named 'result'

while True:
    flag, frame = cap.read()
    frame = cv.flip(frame, 1)
    cv.imshow('result', frame)

    if not flag:
        break

    face_detect_demo(frame)

    if cv.waitKey(1) == ord('q'):  # Modify the keyboard input wait parameter
        break

cv.destroyAllWindows()
cap.release()
