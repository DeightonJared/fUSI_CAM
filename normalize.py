import numpy as np

def normalize_fUSI(images):
    '''
    Return percent change with respect to final two minutes of baseline recording (t in [180-300])
    input/output shape (3600 x 91 x 128)
    '''
    
    norm_images = np.zeros(images.shape)
    
    baseline = images[180:300,:,:] 
    
    ave = np.mean(baseline, axis = 0) + 1e-10
    
    for i in range(images.shape[0]):
        norm_images[i, :,:] = (images[i,:,:] - ave) * 1/ave
        
    return norm_images


def normalize_fUSI_bunch(images):
    '''
    Apply normalization function to multiple mice. Input/output shape (n_mice x 3600 x 91 x 128)
    '''
    
    norm_images = np.zeros(images.shape)
    
    for i in range(images.shape[0]):
        
        norm_images[i] = normalize_fUSI(images[i])
        
    return norm_images