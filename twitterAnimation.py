#twitterAnimation.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
# initialization function: plot the background of each frame


def animate(i):
    pullData = open("twitter-watermelon-out.txt","r").read()
    lines = pullData.split('\n')

    xar = []
    yar = []

    x = 0
    y = 0

    for l in lines:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 1

        xar.append(x)
        yar.append(y)
        
    ax1.clear()
    ax1.plot(xar,yar)

ani = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=20, blit=True)
ani.save('./twitterAnimWaltermelon.gif', writer='imagemagick', fps=600) #imagemagick   ffmpeg

plt.show()
