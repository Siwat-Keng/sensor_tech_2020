from pyrplidar import RPLidar
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def update_line(num, iterator, line):
    scan = next(iterator)
    x = np.array([mea[0] for mea in scan])
    y = np.array([mea[1] for mea in scan])
    line.set_data(x, y)
    return line,    

def run():
    rplidar = RPLidar()
    fig = plt.figure()
    ax = plt.axes(xlim=(-8000, 8000), ylim=(-8000, 8000))
    line, = ax.plot([], [], lw=1)    
    ax.grid(True)

    iterator = rplidar.iter_scan_points()
    ani = animation.FuncAnimation(fig, update_line,
        fargs=(iterator, line,), interval=50)
    plt.show()
    rplidar.stop()
    rplidar.stop_motor()   
    rplidar.disconnect()

if __name__ == '__main__':
    run()