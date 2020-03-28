from pyrplidar import RPLidar
import csv

def run():
    data = []
    rplidar = RPLidar()
    ANGLE = 0
    # ANGLE = 90
    # ANGLE = 180
    # ANGLE = 270
    for distance in rplidar.get_distance(ANGLE):
        data.append(distance)
        if len(data) == 10:
            break

    with open('{}_distance.csv'.format(ANGLE), 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)


    rplidar.stop()
    rplidar.stop_motor()
    rplidar.clear_input()    
    rplidar.reset()
    rplidar.disconnect()

if __name__ == '__main__':
    run()