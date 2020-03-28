from pyrplidar import RPLidar
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

PORT_NAME = 'COM3'
DMAX = 4000
IMIN = 0
IMAX = 50

def update_line(num, iterator, line):
    scan = next(iterator)
    x = np.array([mea[0] for mea in scan])
    y = np.array([mea[1] for mea in scan])
    line.set_data(x, y)
    return line,    

def run():
    lidar = RPLidar(PORT_NAME)
    fig = plt.figure()
    ax = plt.axes(xlim=(-8000, 8000), ylim=(-8000, 8000))
    line, = ax.plot([], [], lw=3)    
    ax.grid(True)

    iterator = lidar.iter_scans_point()
    ani = animation.FuncAnimation(fig, update_line,
        fargs=(iterator, line), interval=50)
    plt.show()
    lidar.stop()
    lidar.disconnect()

if __name__ == '__main__':
    run()