import numpy as np
import cv2 as cv
import glob
from datetime import datetime

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')
result = []

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    drawing = img.copy()

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # # Draw the corners
        # cv.drawChessboardCorners(drawing, (7,6), corners2, ret)
        # cv.imshow("Drawing", drawing)
        # cv.waitKey(50000)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


for img in images:

    frame = cv.imread(img)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    h,  w = frame.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    result.append(dst)

# tot_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     tot_error += error
# print(tot_error/len(objpoints))

for i in range(len(result)):
    cv.imwrite(str(i)+'.png', result[i])

cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np
# import glob

# # termination criteria
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.

# images = glob.glob('*.jpg')
# result = []

# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     drawing = img.copy()

#     ret, corners = cv.findChessboardCorners(gray, (7,6), None)

#     if ret == True:

#         corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

#         objpoints.append(objp)
#         imgpoints.append(corners2)

# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# cap = cv.VideoCapture(0)

# while True:

#     ret, frame = cap.read()

#     cv.imshow("Before", frame)

#     h,  w = frame.shape[:2]
#     newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

#     # undistort
#     dst = cv.undistort(frame, mtx, dist, None, newcameramtx)

#     # crop the image
#     x,y,w,h = roi
#     dst = dst[y:y+h, x:x+w]
#     cv.imshow("After", dst)

#     cv.waitKey(15)