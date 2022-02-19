import matplotlib.animation as animation
from matplotlib import style
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')  # this line is mandatory

fig1 = plt.figure() # in  this the live plot will be saved

ax1 = fig1.add_subplot(1,1,1,)

def animate(p):
    plot_data = open('live_plot_dataset.txt', 'r').read()
    line_data = plot_data.split('\n')
    x1 = [ ]
    y1 = [ ]
    for line in line_data:
        if len(line)>1:
            x,y = line.split(',')
            x1.append(x)
            y1.append(y)

        ax1.clear()
        ax1.plot(x1,y1)

anime_data =  animation.FuncAnimation(fig1, animate,  interval = 100) #interval is in milisecond

plt.show()
