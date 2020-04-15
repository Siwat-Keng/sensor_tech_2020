from pyrplidar import RPLidar
from datetime import datetime
import numpy as np
import cv2, glob

def collect_map(port='COM3', repeat=100):
    rplidar = RPLidar(port)
    result = []
    iterator = rplidar.iter_scans()
    for counter, scan in enumerate(iterator):
        result += scan
        if counter == repeat:
            rplidar.stop()
            rplidar.stop_motor()   
            rplidar.disconnect()            
            break
    with open('{}.npy'.format(datetime.now().timestamp()), 'wb') as filename:
        np.save(filename, np.array(result))

def convert_map(resulution=0.001, threshold=1):
    mappings = glob.glob('*.npy')
    for mapping in mappings:
        with open(mapping, 'rb') as filename:
            _map = np.load(filename, allow_pickle=True)
            _image = np.zeros((int(16/resulution), int(16/resulution),3),np.uint8)
            for scan in _map:
                x = scan['distance']*np.cos(np.radians(scan['angle']))*0.001//resulution
                y = scan['distance']*np.sin(np.radians(scan['angle']))*0.001//resulution
                _image[int(x+8/resulution), int(y+8/resulution)] += 1
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            ret, _image = cv2.threshold(_image, threshold, 255, cv2.THRESH_BINARY)
            cv2.imwrite('{}.png'.format(datetime.now().timestamp()), _image)