import cv2 as cv
from datetime import datetime

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,7), None)

    cv.imshow('img', frame)
    if ret:
        cv.imwrite(str(datetime.now().timestamp())+'.jpg', frame)

    cv.waitKey(1000)