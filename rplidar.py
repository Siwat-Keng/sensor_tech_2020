from rplidar import RPLidar
lidar = RPLidar('COM3')

info = lidar.get_info()
print(info)

health = lidar.get_health()
print(health)

while True:
    new_scan, quality, angle, distance = lidar.iter_mesurments()
    if new_scan:
        print(distance)

lidar.stop()
lidar.stop_motor()
lidar.disconnect()
