import cv2 as cv
import numpy as np


# 应用马赛克效果于图像区域的函数
def apply_mosaic(img, x, y, w, h, block_size):
    # 提取感兴趣区域
    roi = img[y:y + h, x:x + w]

    # 使用块大小调整区域大小以实现马赛克效果
    small_roi = cv.resize(roi, (block_size, block_size))

    # 将小区域调整回原始大小
    mosaic_roi = cv.resize(small_roi, (w, h), interpolation=cv.INTER_NEAREST)

    # 用马赛克效果替换人脸区域
    img[y:y + h, x:x + w] = mosaic_roi


def face_detect_demo(img):
    # 将图像转换为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 加载人脸级联分类器
    face_detector = cv.CascadeClassifier('D:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

    # 在灰度图像中检测人脸
    faces = face_detector.detectMultiScale(gray, 1.25)
    face_count = len(faces)  # 统计检测到的人脸数量

    mosaic_block_size = 20  # 修改此值以获得所需的马赛克效果大小

    for x, y, w, h in faces:
        # 对人脸区域应用马赛克效果
        apply_mosaic(img, x, y, w, h, mosaic_block_size)

    # 绘制矩形框并显示人脸数量
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)

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
