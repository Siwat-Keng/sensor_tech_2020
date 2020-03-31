from pyrplidar import RPLidar
import numpy as np
import cv2, time

time.sleep(10)
rplidar = RPLidar()
mapping = np.zeros((16000,16000,3),np.uint8)
iterator = rplidar.iter_scan_points()

for i in range(1000):
    scan = next(iterator)
    for i in scan:
        mapping[int(i[1]+8000),int(i[0]+8000)] += 1

rplidar.stop()
rplidar.stop_motor()   
rplidar.disconnect()
 
mapping = mapping[4000:12000,4000:12000]
mapping = cv2.cvtColor(mapping, cv2.COLOR_BGR2GRAY)
ret, mapping = cv2.threshold(mapping, 10, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(mapping, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(mapping, contours, -1, (255,255,255), 5)
cv2.imwrite('map.png', mapping)