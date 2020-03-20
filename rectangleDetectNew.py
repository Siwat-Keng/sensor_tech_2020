import cv2 as cv
import numpy as np
from datetime import datetime

def cal(base, res):
    s = np.sum(res, axis=0)
    res = [res[0] / s, res[1] / s, res[2] / s]
    return ((base[0]-res[0])**2+(base[1]-res[1])**2+(base[2]-res[2])**2)**0.5

pink = ([[0.31909547738693467336683417085427, 0.17085427135678391959798994974874, 0.51005025125628140703517587939698], [142, 66, 177]])
purple = np.array([[0.3730407523510971786833855799373, 0.16927899686520376175548589341693, 0.45768025078369905956112852664577],[137, 58, 124]])
dblue = np.array([[0.68911917098445595854922279792746, 0.23834196891191709844559585492228, 0.07253886010362694300518134715026], [109, 43, 16]])
blue = np.array([[0.50671140939597315436241610738255, 0.36577181208053691275167785234899, 0.12751677852348993288590604026846], [141, 92, 30]])
lgreen = np.array([[0.21140939597315436241610738255034, 0.4463087248322147651006711409396, 0.34228187919463087248322147651007], [34, 95, 65]])
forest = np.array([[0.39156627, 0.42168675, 0.18674699], [85, 79, 30]])
dgreen = np.array([[0.37254901960784313725490196078431, 0.44444444444444444444444444444444, 0.18300653594771241830065359477124],[91, 84, 36]])
yellow = np.array([[0.09356725146198830409356725146199, 0.42105263157894736842105263157895, 0.48538011695906432748538011695906], [23, 159, 192]])
orange = np.array([[0.05762711864406779661016949152542,       0.22711864406779661016949152542373, 0.71525423728813559322033898305085], [30, 56, 175]])
red = np.array([[0.10992907801418439716312056737589, 0.17730496453900709219858156028369, 0.71276595744680851063829787234043], [42, 42, 142]])
brown = np.array([[0.12393162393162393162393162393162, 0.25213675213675213675213675213675, 0.62393162393162393162393162393162], [45, 51, 97]])

colors = [pink, purple, dblue, blue, lgreen, yellow, orange, red, brown, forest, dgreen]
text = ["pink", "purple", "darkblue", "blue", "green", "yellow", "orange", "red", "brown", "forest", "darkgreen"]

cap = cv.VideoCapture(0)

while True:

    ret, frame = cap.read()
    image = frame.copy()

    blur = cv.GaussianBlur(frame, (5, 5), 0)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    edge = cv.Canny(gray, 0, 500, apertureSize=5)
    edge = cv.dilate(edge, None)

    contours, _hierachy = cv.findContours(edge.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.02*cv.arcLength(cnt, True), True)
        if len(approx) == 4 and cv.contourArea(approx) > 1000 and cv.isContourConvex(approx) and cv.contourArea(approx) < 100000:
            rectangle = cv.boundingRect(approx)

            if rectangle[2] > 10:
                w = rectangle[2]
                h = rectangle[3]

                if w/h >= 0.8 and w/h <= 1.2:
                    target = []

                    for color in colors:
                        # sm = np.sum(frame[int(rectangle[1])+int(h/2)][int(rectangle[0])+int(w/2)], axis=0)

                        # temp = [frame[int(rectangle[1])+int(h/2)][int(rectangle[0])+int(w/2)][0]/sm, 
                        # frame[int(rectangle[1])+int(h/2)][int(rectangle[0])+int(w/2)][1]/sm, 
                        # frame[int(rectangle[1])+int(h/2)][int(rectangle[0])+int(w/2)][2]/sm]

                        # target.append(((color[0][0]-frame[int(rectangle[1])+int(h/2)][int(rectangle[0])+int(w/2)][0])**2 \
                        #     +(color[0][1]-frame[int(rectangle[1])+int(h/2)][int(rectangle[0])+int(w/2)][1])**2+(color[0][2] \
                        #         -frame[int(rectangle[1])+int(h/2)][int(rectangle[0])+int(w/2)][2])**2)**0.5)
                        target.append(cal(color[0], frame[int(rectangle[1])+int(h/2)][int(rectangle[0])+int(w/2)]))

                    index = target.index(min(target))

                    cv.putText(image, text[index], (int(rectangle[0]), int(rectangle[1])), cv.FONT_HERSHEY_SIMPLEX, 2, colors[index][1], 2, cv.LINE_AA)
                    cv.rectangle(image, (int(rectangle[0]), int(rectangle[1])),(int(rectangle[0] + rectangle[2]), int(rectangle[1] + rectangle[3])),colors[index][1], -1)

                    # cv.putText(frame, text[index], (int(rectangle[0]), int(rectangle[1])), 
                    # cv.FONT_HERSHEY_SIMPLEX, 2, colors[index][1], 2, cv.LINE_AA)
                    # cv.rectangle(frame, (int(rectangle[0]), int(rectangle[1])), 
                    # (int(rectangle[0])+int(rectangle[2]), int(rectangle[1])+int(rectangle[3])), colors[index][1], -1)

    cv.imshow('Frame', frame)
    cv.imshow('Image', image)

    if cv.waitKey(1) & 0xFF == ord(' '):
        cv.imwrite(str(datetime.now().timestamp())+'.jpg', cv.vconcat([frame, image]))    