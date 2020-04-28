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
                try:
                    x = scan['distance']*np.cos(np.radians(scan['angle']))*0.001/resulution
                    y = scan['distance']*np.sin(np.radians(scan['angle']))*0.001/resulution  
                    _image[int(x+8/resulution), int(y+8/resulution)] += 1
                except KeyError:              
                    for angle in scan:
                        x = scan[angle]*np.cos(np.radians(angle))*0.001/resulution
                        y = scan[angle]*np.sin(np.radians(angle))*0.001/resulution
                        _image[int(x+8/resulution), int(y+8/resulution)] += 1
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            ret, _image = cv2.threshold(_image, threshold, 255, cv2.THRESH_BINARY)
            cv2.imwrite('{}.png'.format(mapping.replace('.npy','')), _image)

def png_to_matplotlib(img):
    output = [[],[]]
    for x,v_x in enumerate(img):
        for y,v_y in enumerate(v_x):
            if sum(v_y) != 0:
                output[0].append(x)
                output[1].append(y) 
    return numpy.array(output)