import PySimpleGUI as sg
import cv2

layout = [
    [sg.Image(key='-TMAGE-')],
    [sg.Image(filename='7.png')],
    [sg.Text('People in picture: 0', key='-TEXT-', expand_x=True, justification='c')],
    [sg.Button('停止', key='-STOP-')]
]
window = sg.Window('Face Detection', layout)
# 获取视频
video = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('D:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

while True:
    event, values = window.read(timeout=0)
    if event == sg.WIN_CLOSED:
        break
    flag, frame = video.read()
    # 翻转帧
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(50, 50))
    # 绘制矩形框
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('result', frame)
    # 更新图像
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    # window['-TMAGE-'].update(data=imgbytes)
    # 更新文本
    window['-TEXT-'].update(f'People in picture: {len(faces)}')
    if event == '-STOP-':
        break
window.close()
