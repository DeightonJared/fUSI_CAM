import numpy as np
import matplotlib.pyplot as plt
import torch
from torchcam.methods import XGradCAM
import torch.nn.functional as F
from matplotlib import animation
from tqdm import tqdm
import scipy.io as sio

reference = sio.loadmat('Reference_mkdose.mat')['Reference']

def mouse_movie(dataset, mouse_index):
    '''
    dataset should be [mice x time x 91 x 128]
    '''
    
    video = dataset[mouse_index]
    fig = plt.figure()
    a = video[0]
    # im = plt.imshow(a, interpolation='none', vmin=-5, vmax=5)
    im = plt.imshow(a, interpolation='none', cmap = 'grey')
    plt.colorbar()
    
    def animate(i):
        arr = im.get_array()
        arr =  video[i]  # exponential decay of the values
        im.set_array(arr)
        return [im]
    
    
    anim = animation.FuncAnimation(fig, animate,
                                    frames=tqdm(range(video.shape[0])), interval=10, blit=True)
    
    plt.show()
    anim.save('eg_anim' + '.gif', writer='pillow', fps=30)
    

def showImgs(imgs, n_imgs, i_imgs, labels=None):
    '''
    Show n_imgs of images form imgs, indexed by i_imgs.
    
    Parameters:
    - imgs: Image data
    - n_imgs: Number of images to show
    - i_imgs: Indices of the images to show
    - labels: Optional labels for each image (default is None)
    
    '''
    n = np.sqrt(n_imgs)
    m = n
    p = 1
    if n != int(n):
       n = round(n)
       m = n + 1

    fig = plt.figure()
    for idx, i in enumerate(i_imgs):
       ax = fig.add_subplot(int(n), int(m), p)
       
       plt.imshow(imgs[i], cmap='grey')
       plt.axis('off')
       
       # If labels are provided, show them below the image
       if labels is not None:
           ax.text(0.5, -0.3, str(round(labels[idx],2)), size=10, ha="center", transform=ax.transAxes)
       
       p += 1
    return fig, ax

def showImg(img):
    ''' show a single image '''
    showImgs([img], 1, [0])
    
    
def get_CAMs(model, test_dataset, mk_num_images, sal_num_images, cam_extractor = None):
    
    image_size = test_dataset.__getitem__(0)[0].shape
    
    # Initial placeholders for accumulated CAM and images
    mk_accumulated_cam = torch.zeros(image_size)
    mk_accumulated_img = torch.zeros(image_size)  # Assuming img is 2D
    
    for idx in range(mk_num_images):
        input_tensor, label = test_dataset.__getitem__(idx)
        mk_accumulated_img += input_tensor # or usual_data.__getitem__(idx)[0] for non-normalizede
    
        input_tensor = input_tensor.unsqueeze(0)
        model.eval()
        out = model(input_tensor.unsqueeze(1))
        if torch.isnan(out).any():
            print(f"NaN value in model output at idx {idx}")
        prediction = out.squeeze(0).argmax().item()
    
        activation_map = cam_extractor(prediction, out)
        if torch.isnan(activation_map[0]).any():
            print(f"NaN value in activation_map at idx {idx}")
        upsampled_cam = F.interpolate(activation_map[0].unsqueeze(0), size=(image_size[0], image_size[1]), mode='bilinear', align_corners=True)
        
        mk_accumulated_cam += upsampled_cam.squeeze(0).squeeze(0)
    
    # Compute the average CAM and image
    mk_avg_cam = mk_accumulated_cam / mk_num_images
    mk_avg_img = mk_accumulated_img / mk_num_images

    # Now saline

    # Initial placeholders for accumulated CAM and images
    sal_accumulated_cam = torch.zeros(image_size)
    sal_accumulated_img = torch.zeros(image_size)  # Assuming img is 2D
    
    for idx in range(mk_num_images, mk_num_images + sal_num_images):
        input_tensor, label = test_dataset.__getitem__(idx)
        sal_accumulated_img += input_tensor # or usual_data.__getitem__(idx)[0] for non-normalizede
    
        input_tensor = input_tensor.unsqueeze(0)
        model.eval()
        out = model(input_tensor.unsqueeze(1))
        if torch.isnan(out).any():
            print(f"NaN value in model output at idx {idx}")
        prediction = out.squeeze(0).argmax().item()
    
        activation_map = cam_extractor(prediction, out)
        if torch.isnan(activation_map[0]).any():
            print(f"NaN value in activation_map at idx {idx}")
        
        
        upsampled_cam = F.interpolate(activation_map[0].unsqueeze(0), size=(image_size[0], image_size[1]), mode='bilinear', align_corners = True)
    
        sal_accumulated_cam += upsampled_cam.squeeze(0).squeeze(0)
    
    # Compute the average CAM and image
    sal_avg_cam = sal_accumulated_cam / sal_num_images
    sal_avg_img = sal_accumulated_img / sal_num_images
    
    
    return mk_avg_cam, sal_avg_cam, mk_avg_img, sal_avg_img


def get_CAMs_anasthesia(model, test_dataset, none_num_images, sal_num_images, cam_extractor = None):
    
    image_size = test_dataset.__getitem__(0)[0].shape
    
    # Initial placeholders for accumulated CAM and images
    none_accumulated_cam = torch.zeros(image_size)
    none_accumulated_img = torch.zeros(image_size)  # Assuming img is 2D
    
    for idx in range(none_num_images):
        input_tensor, label = test_dataset.__getitem__(idx)
        none_accumulated_img += input_tensor # or usual_data.__getitem__(idx)[0] for non-normalizede
    
        input_tensor = input_tensor.unsqueeze(0)
        model.eval()
        out = model(input_tensor.unsqueeze(1))
        if torch.isnan(out).any():
            print(f"NaN value in model output at idx {idx}")
        prediction = out.squeeze(0).argmax().item()
    
        activation_map = cam_extractor(prediction, out)
        if torch.isnan(activation_map[0]).any():
            print(f"NaN value in activation_map at idx {idx}")
        upsampled_cam = F.interpolate(activation_map[0].unsqueeze(0), size=(image_size[0], image_size[1]), mode='bilinear', align_corners=True)
        
        none_accumulated_cam += upsampled_cam.squeeze(0).squeeze(0)
    
    # Compute the average CAM and image
    none_avg_cam = none_accumulated_cam / none_num_images
    none_avg_img = none_accumulated_img / none_num_images

    # Now saline

    # Initial placeholders for accumulated CAM and images
    sal_accumulated_cam = torch.zeros(image_size)
    sal_accumulated_img = torch.zeros(image_size)  # Assuming img is 2D
    
    for idx in range(none_num_images, none_num_images + sal_num_images):
        input_tensor, label = test_dataset.__getitem__(idx)
        sal_accumulated_img += input_tensor # or usual_data.__getitem__(idx)[0] for non-normalizede
    
        input_tensor = input_tensor.unsqueeze(0)
        model.eval()
        out = model(input_tensor.unsqueeze(1))
        if torch.isnan(out).any():
            print(f"NaN value in model output at idx {idx}")
        prediction = out.squeeze(0).argmax().item()
    
        activation_map = cam_extractor(prediction, out)
        if torch.isnan(activation_map[0]).any():
            print(f"NaN value in activation_map at idx {idx}")
        upsampled_cam = F.interpolate(activation_map[0].unsqueeze(0), size=(image_size[0], image_size[1]), mode='bilinear', align_corners = True)
        
        sal_accumulated_cam += upsampled_cam.squeeze(0).squeeze(0)
    
    # Compute the average CAM and image
    sal_avg_cam = sal_accumulated_cam / sal_num_images
    sal_avg_img = sal_accumulated_img / sal_num_images
    
    
    return none_avg_cam, sal_avg_cam, none_avg_img, sal_avg_img

def plot_CAMs(mk_avg_cam, sal_avg_cam, cbar_min_max=None, threshold = 0.15, savename='Combined_CAMs', save=True):
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(1, 3)  # 1 row, 3 columns
    
    diff_vmin, diff_vmax = cbar_min_max if cbar_min_max else (None, None)

    # Plot MK average CAM
    axes[0].imshow(reference, cmap = 'grey')
    axes[0].imshow(mk_avg_cam, cmap='viridis',  alpha = 0.5)
    axes[0].set_title("MK Average CAM")
   

    # Plot Saline average CAM
    axes[1].imshow(reference, cmap = 'grey')
    axes[1].imshow(sal_avg_cam, cmap='viridis',  alpha = 0.5)
    axes[1].set_title("Saline Average CAM")
    

    # Plot the difference between MK and Saline CAMs
    diff = mk_avg_cam - sal_avg_cam
    diff = np.ma.masked_where(diff < threshold, diff)
    axes[2].imshow(reference, cmap = 'grey')
    
    im_diff = axes[2].imshow(diff, cmap='plasma', alpha = 0.5, vmin=diff_vmin, vmax=diff_vmax)
    axes[2].set_title('Difference')
    
    cbar = plt.colorbar(im_diff, ax=axes[2], shrink=0.5, aspect=20)

    plt.tight_layout()
    if save:
        plt.savefig(f'{savename}.png')
    plt.show()
    
def plot_CAMs_highdose_lowdose(mk15_avg_cam, mk01_avg_cam, cbar_min_max=None, threshold = 0.15, savename='Combined_CAMs', save=True):
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(1, 3)  # 1 row, 3 columns
    
    diff_vmin, diff_vmax = cbar_min_max if cbar_min_max else (None, None)

    # Plot MK average CAM
    axes[0].imshow(reference, cmap = 'grey')
    axes[0].imshow(mk15_avg_cam, cmap='viridis',  alpha = 0.5)
    axes[0].set_title("MK1_5 Average CAM")
   

    # Plot Saline average CAM
    axes[1].imshow(reference, cmap = 'grey')
    axes[1].imshow(mk01_avg_cam, cmap='viridis',  alpha = 0.5)
    axes[1].set_title("MK0_1 Average CAM")
    

    # Plot the difference between MK and Saline CAMs
    diff = mk15_avg_cam - mk01_avg_cam
    diff = np.ma.masked_where(diff < threshold, diff)
    axes[2].imshow(reference, cmap = 'grey')
    
    im_diff = axes[2].imshow(diff, cmap='plasma', alpha = 0.5, vmin=diff_vmin, vmax=diff_vmax)
    axes[2].set_title('Difference')
    
    cbar = plt.colorbar(im_diff, ax=axes[2], shrink=0.5, aspect=20)

    plt.tight_layout()
    if save:
        plt.savefig(f'{savename}.png')
    plt.show()
    
def plot_CAMs_anasthesia(none_avg_cam, sal_avg_cam, cbar_min_max=None, threshold = 0.15, savename='Combined_CAMs', save=True):
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(1, 3)  # 1 row, 3 columns
    
    diff_vmin, diff_vmax = cbar_min_max if cbar_min_max else (None, None)

    # Plot Saline average CAM
    axes[1].imshow(reference, cmap = 'grey')
    axes[1].imshow(none_avg_cam, cmap='viridis',  alpha = 0.5)
    axes[1].set_title("None Average CAM")
   

    # Plot None average CAM
    axes[0].imshow(reference, cmap = 'grey')
    axes[0].imshow(sal_avg_cam, cmap='viridis',  alpha = 0.5)
    axes[0].set_title("Saline Average CAM")
    

    # Plot the difference between MK and Saline CAMs
    diff = sal_avg_cam - none_avg_cam 
    diff = np.ma.masked_where(diff < threshold, diff)
    axes[2].imshow(reference, cmap = 'grey')
    
    im_diff = axes[2].imshow(diff, cmap='plasma', alpha = 0.5, vmin=diff_vmin, vmax=diff_vmax)
    axes[2].set_title('Difference')
    
    cbar = plt.colorbar(im_diff, ax=axes[2], shrink=0.5, aspect=20)

    plt.tight_layout()
    if save:
        plt.savefig(f'{savename}.png')
    plt.show()
    
    
def showImgCAM(model, dataset, idx, cam_extractor = None, unnorm_data = None, threshold = 0.15, upsample = True, savename = 'savename', save = False):
        
    input_tensor, label = dataset.__getitem__(idx)
    image_size = input_tensor.shape
    if unnorm_data is not None:
        img = unnorm_data.__getitem__(idx)[0]
    else:
        img = input_tensor

    input_tensor = input_tensor.unsqueeze(0)
    
    out = model(input_tensor.unsqueeze(1))
    
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    if upsample:
        upsampled_cam = F.interpolate(activation_map[0].unsqueeze(0), size=(image_size[0], image_size[1]), mode='bilinear', align_corners=True)
        cam = upsampled_cam.squeeze(0).squeeze(0)
        cam = np.ma.masked_where(cam < threshold, cam)
    else:
        cam = activation_map[0].squeeze(0)
        cam = np.ma.masked_where(cam < threshold, cam)
        
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap = 'grey')  # Assuming the image is grayscale
    axes[0].set_title("Image")
    
    axes[1].imshow(reference, cmap = 'grey')
    axes[1].imshow(cam, alpha = 0.6)  # CAM is typically visualized with viridis cmap
    axes[1].set_title("CAM")

    plt.tight_layout()
    if save:
        plt.savefig(savename + '.png')
    plt.show()
    
def CAM_movie(mk_avg_CAMs, sal_avg_CAMs, savename = 'eg_movie', threshold = 0.15):

    def update(i):
        ax1.imshow(mk_avg_CAMs[i], cmap='viridis')
        ax2.imshow(sal_avg_CAMs[i], cmap='viridis')
        
        diff = mk_avg_CAMs[i] - sal_avg_CAMs[i]
        diff = np.ma.masked_where(diff < threshold, diff)
        
        ax3.imshow(reference, cmap = 'grey')
        ax3.imshow(diff, cmap='plasma', alpha = 0.5)
        
        plt.suptitle(f"Time Step: {i}")


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.set_title('MK Average CAMs')
    ax2.set_title('SAL Average CAMs')
    ax3.set_title('Difference')

    #
    ani = animation.FuncAnimation(fig, update, frames=len(mk_avg_CAMs), blit=False)


    ani.save(savename + '.gif', writer='pillow', fps=4)
    
def CAM_movie_cropped(mk_avg_CAMs, sal_avg_CAMs, savename = 'eg_movie', threshold = 0.15):

    def update(i):
        ax1.imshow(mk_avg_CAMs[i], cmap='viridis')
        ax2.imshow(sal_avg_CAMs[i], cmap='viridis')
        
        diff = mk_avg_CAMs[i] - sal_avg_CAMs[i]
        diff = np.ma.masked_where(diff < threshold, diff)
        
        ax3.imshow(reference[7:78,25:109], cmap = 'grey') # Todo: this can change depending on which cropped we are using, reference[7:81,:] or reference[7:78,25:109] or reference[1:92, 25:109]
        ax3.imshow(diff, cmap='plasma', alpha = 0.5)
        
        plt.suptitle(f"Time Step: {i}")


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.set_title('MK Average CAMs')
    ax2.set_title('SAL Average CAMs')
    ax3.set_title('Difference')

    #
    ani = animation.FuncAnimation(fig, update, frames=len(mk_avg_CAMs), blit=False)


    ani.save(savename + '.gif', writer='pillow', fps=4)
    
def CAM_movie_anasthesia(none_avg_CAMs, sal_avg_CAMs, savename = 'eg_movie', threshold = 0.15):

    def update(i):
        ax1.imshow(sal_avg_CAMs[i], cmap='viridis')
        ax2.imshow(none_avg_CAMs[i], cmap='viridis')
        
        diff = sal_avg_CAMs[i] - none_avg_CAMs[i] 
        diff = np.ma.masked_where(diff < threshold, diff)
        
        ax3.imshow(reference, cmap = 'grey')
        ax3.imshow(diff, cmap='plasma', alpha = 0.5)
        
        plt.suptitle(f"Time Step: {i}")


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.set_title('Saline Average CAMs')
    ax2.set_title('None Average CAMs')
    ax3.set_title('Difference')

    #
    ani = animation.FuncAnimation(fig, update, frames=len(none_avg_CAMs), blit=False)


    ani.save(savename + '.gif', writer='pillow', fps=4)

def CAM_movie_MK(mk_avg_CAMs, savename = 'eg_movie', threshold = 0.15):

    def update(i):
        mk_mask = np.ma.masked_where(mk_avg_CAMs[i] < threshold, mk_avg_CAMs[i])
        
        ax1.imshow(reference, cmap = 'grey')
        ax1.imshow(mk_mask, cmap='plasma', alpha = 0.5)
        
       
        plt.suptitle(f"Time Step: {i}")


    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.set_title('MK Average CAMs')
    

    #
    ani = animation.FuncAnimation(fig, update, frames=len(mk_avg_CAMs), blit=False)


    ani.save(savename + '.gif', writer='pillow', fps=4)
    
    
def CAM_movie_Sal(sal_avg_CAMs, savename = 'eg_movie', threshold = 0.15):

    def update(i):
        sal_mask = np.ma.masked_where(sal_avg_CAMs[i] < threshold, sal_avg_CAMs[i])
        
        ax1.imshow(reference, cmap = 'grey')
        ax1.imshow(sal_mask, cmap='plasma', alpha = 0.5)
        
       
        plt.suptitle(f"Time Step: {i}")


    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.set_title('Saline Average CAMs')
    

    #
    ani = animation.FuncAnimation(fig, update, frames=len(sal_avg_CAMs), blit=False)


    ani.save(savename + '.gif', writer='pillow', fps=4)
    
def CAM_movie_None(none_avg_CAMs, savename = 'eg_movie', threshold = 0.15):

    def update(i):
        none_mask = np.ma.masked_where(none_avg_CAMs[i] < threshold, none_avg_CAMs[i])
        
        ax1.imshow(reference, cmap = 'grey')
        ax1.imshow(none_mask, cmap='plasma', alpha = 0.5)
        
       
        plt.suptitle(f"Time Step: {i}")


    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.set_title('None Average CAMs')
    

    #
    ani = animation.FuncAnimation(fig, update, frames=len(none_avg_CAMs), blit=False)


    ani.save(savename + '.gif', writer='pillow', fps=4)
    
def CAM_movie_MK_cropped(mk_avg_CAMs, savename = 'eg_movie', threshold = 0.15):

    def update(i):
        mk_mask = np.ma.masked_where(mk_avg_CAMs[i] < threshold, mk_avg_CAMs[i])
        
        ax1.imshow(reference[7:78,25:109], cmap = 'grey')
        ax1.imshow(mk_mask, cmap='plasma', alpha = 0.5)
        
       
        plt.suptitle(f"Time Step: {i}")


    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.set_title('MK Average CAMs')
    

    #
    ani = animation.FuncAnimation(fig, update, frames=len(mk_avg_CAMs), blit=False)


    ani.save(savename + '.gif', writer='pillow', fps=4)

