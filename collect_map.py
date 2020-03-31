from pyrplidar import RPLidar
import numpy as np
import cv2, time, winsound

time.sleep(10) # 10 sec run away!!!

rplidar = RPLidar() #default COM3
mapping = np.zeros((16000,16000,3),np.uint8)
iterator = rplidar.iter_scan_points()

for i in range(8000):
    scan = next(iterator)
    for i in scan:
        mapping[int(i[1]+8000),int(i[0]+8000)] += 1

rplidar.stop()
rplidar.stop_motor()   
rplidar.disconnect()
 
mapping = mapping[4000:12000,4000:12000] #crop
mapping = cv2.cvtColor(mapping, cv2.COLOR_BGR2GRAY) #format picture
ret, mapping = cv2.threshold(mapping, 3, 255, cv2.THRESH_BINARY) #remove noise

#help? not sure
contours, hierarchy = cv2.findContours(mapping, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
cv2.drawContours(mapping, contours, -1, (255,255,255), 5)

cv2.imwrite('map.png', mapping) #save image

#beep when finished
while True:
    winsound.Beep(1500, 500)
    cv2.waitKey(100)