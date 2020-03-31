import numpy as np
import cv2 as cv

# colorLower_1 = (10, 190, 50)
# colorUpper_1 = (20,255,200)

#dark
colorLower_1 = (15,100,50)
colorUpper_1 = (32,255,255)

# colorLower_2 = (15, 145, 80)
# colorUpper_2 = (21, 255, 255)

# 2x fluorescent(night)
# colorLower_2 = (15, 15, 250)
# colorUpper_2 = (35, 250, 255)
colorLower_2 = (12, 110, 175)
colorUpper_2 = (25, 255, 255)


cap = cv.VideoCapture(0)

while True:
	ret, frame = cap.read()
	frame = cv.resize(frame, (600,480))	

	img = cv.medianBlur(frame, 3)
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

	first = cv.inRange(hsv, colorLower_1, colorUpper_1)
	second = cv.inRange(hsv, colorLower_2, colorUpper_2)
	img = cv.addWeighted( src1=first, alpha=1, src2=second, beta=1, gamma=0, dst=img)

	img = cv.erode(img, None, iterations=2)
	img = cv.dilate(img, None, iterations=2)
	img = cv.GaussianBlur(img, (9,9), 2)	

	circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1,200,
                                    param1=50,param2=30,minRadius=10)	

	if circles is not None:
		circles = np.round(circles[0, :]).astype("int")
		for (x, y, r) in circles:
			cv.circle(frame,(int(x), int(y)), int(r),
							(0, 0, 255), 2)

	#cv.imshow('img', img)
	#cv.imshow('first', first)
	#cv.imshow('second', second)
	cv.imshow('frame', frame)
	
	cv.waitKey(1)

