import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

mk1_5 = np.moveaxis(np.load('MK801_Exp/mk1_5_Afterregis.npy'), 3,1)
sal= np.moveaxis(np.load('MK801_Exp/sal_Afterregis.npy'), 3,1)
print(mk1_5.shape, sal.shape)

num_mice = 5
num_time_steps = 180


fig, axes = plt.subplots(nrows=num_mice, ncols=2, figsize=(10, 25))
axes[0, 0].set_title('MK', fontsize = 24)
axes[0, 1].set_title('Saline', fontsize = 24)

for ax in axes.flat:
        ax.axis('off')
        
ims = []
for i in range(num_mice):
    im_mk = axes[i, 0].imshow(mk1_5[0,0,:,:], cmap='gray', interpolation='none')
    im_sal = axes[i, 1].imshow(sal[0,0,:,:], cmap='gray', interpolation='none')
    ims.append([im_mk, im_sal])

def animate(i):
    for row in range(num_mice):
        ims[row][0].set_array(mk1_5[row, i])
        ims[row][1].set_array(sal[row, i])
        fig.suptitle(f"Time Step: {i}", fontsize = 24)
    return [im for sublist in ims for im in sublist]  # Flatten the list

anim = FuncAnimation(fig, animate, frames=range(num_time_steps), interval=10, blit=False)

plt.tight_layout()
plt.close(fig)  # Close the figure to prevent it from showing inline in notebooks
anim.save('nonrigid_alignment_mouse_movie' + '_' + str(num_mice) + '.gif', writer='pillow', fps=10)


