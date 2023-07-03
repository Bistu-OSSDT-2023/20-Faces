import cv2 as cv
import numpy as np


# Function to apply mosaic effect on an image region
def apply_mosaic(img, x, y, w, h, block_size):
    # Extract the region of interest
    roi = img[y:y + h, x:x + w]

    # Resize the region using block size for mosaic effect
    small_roi = cv.resize(roi, (block_size, block_size))

    # Resize the small region back to original size
    mosaic_roi = cv.resize(small_roi, (w, h), interpolation=cv.INTER_NEAREST)

    # Replace the face region with the mosaic effect
    img[y:y + h, x:x + w] = mosaic_roi


def face_detect_demo(img):
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Load the face cascade classifier
    face_detector = cv.CascadeClassifier('D:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale image
    faces = face_detector.detectMultiScale(gray, 1.25)
    face_count = len(faces)  # Count the number of faces detected

    mosaic_block_size = 20  # Change this value for desired mosaic effect size

    for x, y, w, h in faces:
        # Apply the mosaic effect to the face region
        apply_mosaic(img, x, y, w, h, mosaic_block_size)

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
