import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as a
import os
import pathlib

fig, ax = plt.subplots(figsize=(5,3))
ax.set(xlim=(-3,3),ylim=(-1,1))



x = np.linspace(-3,3,91)
t = np.linspace(1,25,30)
X2, T2 = np.meshgrid(x,t)

sinT2 = np.sin(2*np.pi*T2/T2.max())
F = 0.9 * sinT2 * np.sinc(X2*(1+sinT2))

line = ax.plot(x, F[0,:], color='k', lw=2)[0]

def animate(i):
    line.set_ydata(F[i,:])

anim = a.FuncAnimation(fig, animate, interval = 100, frames = len(t)-1)
plt.draw()
plt.show()


f = os.path.join(pathlib.Path(__file__).parent.resolve(),'FirstAnimation')
writergif = a.PillowWriter(fps=30)
anim.save(f+'.gif', writer=writergif)
# writer_video = a.FFMpegWriter(fps = 60)
# anim.save(f+'.mp4', writer=writer_video)

# anim.save('FirstGiving.gif', writer='imagemagick')


