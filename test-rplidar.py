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
<<<<<<< HEAD
    x = np.array([mea[0] for mea in scan])
    y = np.array([mea[1] for mea in scan])
    line.set_data(x, y)
    return line,    
=======
    offsets = np.array([(np.radians(meas[1]), meas[2]) for meas in scan])
    line.set_offsets(offsets)
    intens = np.array([meas[0] for meas in scan])
    line.set_array(intens)
    return line,
>>>>>>> 9eef46b18376f058960b7725567cfb8e21a077f0

def run():
    lidar = RPLidar(PORT_NAME)
    fig = plt.figure()
<<<<<<< HEAD
    ax = plt.axes(xlim=(-8000, 8000), ylim=(-8000, 8000))
    line, = ax.plot([], [], lw=3)    
    ax.grid(True)

    iterator = lidar.iter_scans_point()
=======
    ax = plt.subplot(111, projection='polar')
    line = ax.scatter([0, 0], [0, 0], s=5, c=[IMIN, IMAX],
                           cmap=plt.cm.Greys_r, lw=0)
    ax.set_rmax(DMAX)
    ax.grid(True)

    iterator = lidar.iter_scans()
>>>>>>> 9eef46b18376f058960b7725567cfb8e21a077f0
    ani = animation.FuncAnimation(fig, update_line,
        fargs=(iterator, line), interval=50)
    plt.show()
    lidar.stop()
    lidar.disconnect()

if __name__ == '__main__':
    run()