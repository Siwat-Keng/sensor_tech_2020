import numpy as np
import cv2 as cv
import imutils
from datetime import datetime

cap = cv.VideoCapture(0)

white_upper = (180,30,205)
white_lower = (0,0,120)

white_upper1 = (200,250,210)
white_lower1 = (100,100,0)



while True:
    ret, frame = cap.read()

    # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # gray = cv.inRange(hsv, white_lower, white_upper)   
    ret,gray = cv.threshold(frame,127,255,cv.THRESH_BINARY)

    new_frame = np.zeros((600,600),np.uint8)

    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 11, 17, 17)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv.erode(gray,kernel,iterations = 2)
    kernel = np.ones((4,4),np.uint8)   
    dilation = cv.dilate(erosion,kernel,iterations = 2) 
    edged = cv.Canny(dilation, 30, 200)

    contours, hier = cv.findContours(edged, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:

        contours_poly = cv.approxPolyDP(cnt, cv.arcLength(cnt, True), True)

        if len(contours_poly) == 4:
            
            
            (x,y,w,h) = cv.boundingRect(contours_poly)

            new_frame = frame[y:y+h, x:x+w, :]  

            new_frame = cv.resize(new_frame, (600,600))
            hsv = cv.cvtColor(new_frame, cv.COLOR_BGR2HSV)
            white = cv.inRange(new_frame, white_lower1, white_upper1) 
            cv.bitwise_not(new_frame, new_frame, white)
            
            h, w = white.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)

            cv.floodFill(white, mask, (0,0), 255)
            cv.floodFill(white, mask, (0,599), 255)
            cv.floodFill(white, mask, (599,0), 255)
            cv.floodFill(white, mask, (599,599), 255)
            cv.floodFill(white, mask, (300,0), 255)
            cv.floodFill(white, mask, (300,599), 255)
            cv.floodFill(white, mask, (0,300), 255)
            cv.floodFill(white, mask, (599,300), 255)     

            temp = np.zeros((600,600,3),np.uint8)
            temp = cv.bitwise_not(temp)
            cv.bitwise_not(temp, temp, white)


            new_frame = cv.bitwise_and(new_frame, temp)

            edged = cv.Canny(white, 30, 200) 

            cnts = cv.findContours(edged, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            for c in cnts:            

                if 50 < cv.contourArea(c):
                    contours_poly = cv.approxPolyDP(c, 3, True)
                    if len(contours_poly) == 4:
                        (x,y,w,h) = cv.boundingRect(contours_poly)
                        r,g,b = new_frame[y+h//2,x+w//2]
                        if r!=0 or g!=0 or b!=0:
                            cv.rectangle(new_frame, (int(x-10),int(y+10)), (int(x+w+10),int(y+h-10)), (int(r),int(g),int(b)), 3)

    cv.imshow('IMG',new_frame)

    if cv.waitKey(1) & 0xFF == ord(' '):
        cv.imwrite(str(datetime.now().timestamp())+'.jpg', new_frame)

# import cv2 as cv
# import numpy as np
# import imutils, os
# from datetime import datetime

# pink = np.array([[140,130,150], [170,255,255], [155,255,255]])
# purple = np.array([[130,110,140], [140,255,255], [135,255,255]])
# dblue = np.array([[110,130,150], [130,255,255], [120,255,255]])
# blue = np.array([[90,130,150], [110,255,255], [100,255,255]])
# lgreen = np.array([[40,60,20], [75,255,255], [60,255,255]])
# dgreen = np.array([[95,60,10], [105,255,120], [100,255,120]])
# dgreen2 = np.array([[80,60,10], [95,255,255], [88,255,255]])
# yellow = np.array([[20,70,150], [40,255,255], [30,255,255]])
# orange = np.array([[5,70,150], [20,255,255], [10,255,255]])
# red = np.array([[0,70,150], [5,255,255], [0,255,255]])
# red2 = np.array([[175,70,150], [179,255,255], [0,255,255]])
# brown = np.array([[175,70,100], [179,160,200], [0,150,150]])
# brown2 = np.array([[0,100,100], [5,160,190], [0,150,150]])

# colors = [pink, purple, dblue, blue, lgreen, yellow, orange, red, brown, dgreen, dgreen2]




# cap = cv.VideoCapture(0)

# while(1):

#     ret, frame = cap.read()

#     frame = cv.resize(frame,(640,480))

#     drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     gray = cv.bilateralFilter(gray, 11, 17, 17)

#     kernel = np.ones((5,5),np.uint8)
#     erosion = cv.erode(gray,kernel,iterations = 2)
#     kernel = np.ones((4,4),np.uint8)
#     dilation = cv.dilate(erosion,kernel,iterations = 2)

#     edged = cv.Canny(dilation, 30, 200)

#     contours = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     contours = imutils.grab_contours(contours)
    
#     if len(contours) > 0:
#             c = max(contours, key=cv.contourArea)
#             contours_poly = cv.approxPolyDP(c, 3, True)
#             if 5000 < cv.contourArea(c) < 100000:
#                 (x,y,w,h) = cv.boundingRect(contours_poly)
#                 #frame = frame[y:y+h, x:x+w, :]          
#                 frame = cv.resize(frame,(640,480))

#                 frame = cv.GaussianBlur(frame, (7, 7), 0)
#                 hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#                 drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

#                 for e in colors:
#                     lower, upper = e[0:2]
#                     mask = cv.inRange(hsv, lower, upper)
#                     mask = cv.erode(mask, None, iterations=2)
#                     res = cv.bitwise_and(frame,frame, mask= mask)

#                     contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#                     contours = imutils.grab_contours(contours)

#                     if len(contours) > 0:
#                         c = max(contours, key=cv.contourArea)
#                         if cv.contourArea(c) > 500:
#                             contours_poly = cv.approxPolyDP(c, 3, True)
#                             rect = cv.boundingRect(contours_poly)
#                             r,g,b = frame[int(rect[1])+rect[3]//2,int(rect[0])+rect[2]//2]
#                             cv.rectangle(drawing, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (int(r),int(g),int(b)), -1)


#     cv.imshow('frame',frame)
#     cv.imshow('draw', drawing)

#     if cv.waitKey(1) & 0xFF == ord(' '):
#         cv.imwrite(str(datetime.now().timestamp())+'.jpg', drawing)