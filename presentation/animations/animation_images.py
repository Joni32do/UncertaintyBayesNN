import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as a
import os
import pathlib

path_npy = os.path.join(os.path.dirname(__file__),"95.npy")
u = np.load(path_npy)

c = u[:,:,:,0]
sk = u[:,:,:,1]
n_s, n_x, t = np.shape(c)


img = c


width = t
length  = n_x

abs_length = 3
rel_width = abs_length * width/length

fig= plt.figure()
fig.set_size_inches(rel_width, abs_length)
ax = plt.Axes(fig, [0.,0.,1.,1.])
ax.set_axis_off()
fig.add_axes(ax)
# ax.set(xlim=(0,1),ylim=(0,28))


image = ax.imshow(img[0,:,:], cmap = 'viridis')
def animate(i):
    image.set(data = img[i,:,:])

anim = a.FuncAnimation(fig, animate, interval = 1000, frames = n_s-1)

f = os.path.join(pathlib.Path(__file__).parent.resolve(),'Noise95_c')
writergif = a.PillowWriter(fps=10)
anim.save(f+'.gif', writer=writergif)
# writer_video = a.FFMpegWriter(fps = 60)
# anim.save(f+'.mp4', writer=writer_video)

# anim.save('FirstGiving.gif', writer='imagemagick')


