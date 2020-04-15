from pyrplidar import RPLidar
import csv

def run():
    data = []
    rplidar = RPLidar()
    ANGLE = 0
    # ANGLE = 90
    # ANGLE = 180
    # ANGLE = 270
    try:
        for distance in rplidar.get_distance(ANGLE):
            # print(distance)
            data.append(distance)
            if len(data) == 1000:
                break
            elif len(data) % 100 == 0:
                print('Data added : {}'.format(len(data)))

        with open('{}_distance.csv'.format(ANGLE), 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)

    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    rplidar.stop()
    rplidar.stop_motor()
    rplidar.clear_input()    
    rplidar.reset()
    rplidar.disconnect()

if __name__ == '__main__':
    run()