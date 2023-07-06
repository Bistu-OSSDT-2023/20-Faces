import cv2 as cv
import numpy as np


def face_detect_demo(img):
    # 将图像转换为灰度头像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 加载人脸级联分类器
    face_detector = cv.CascadeClassifier('D:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

    # 在灰度图像中检测人脸
    faces = face_detector.detectMultiScale(gray, 1.25)
    face_count = len(faces)  # Count the number of faces detected

    # 加载替换图像
    replace_img = cv.imread('5.png')

    for x, y, w, h in faces:
        # 将人脸区域替换为替换图像
        img[y:y + h, x:x + w] = cv.resize(replace_img, (w, h))

    # 绘制矩形框并显示人脸数量
    for x, y, w, h in faces:
        cv.putText(img, f"Faces: {face_count}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('result', img)


# 连接到网络摄像头
cap = cv.VideoCapture(0)
cv.namedWindow('result')  # 创建名为'结果'的窗口

while True:
    flag, frame = cap.read()
    frame = cv.flip(frame, 1)
    cv.imshow('result', frame)

    if not flag:
        break

    face_detect_demo(frame)

    if cv.waitKey(1) == ord('q'):  # 修改键盘输入等待参数
        break

cv.destroyAllWindows()
cap.release()
