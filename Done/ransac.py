import cv2 as cv
import numpy as np


def getPointPositions(binaryImage):
    pass

cap = cv.VideoCapture(0)

while(1):

    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    minval, maxval, minloc, maxloc = cv.minMaxLoc(gray)
    mul = 255.0/(maxval-minval)
    normalized = gray - minval
    
    _input = cv.threshold(frame, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    norm = cv.threshold(normalized, 100, 255, cv.THRESH_BINARY)
    

    cv.imshow("Frame", frame)
    cv.waitKey(1)



