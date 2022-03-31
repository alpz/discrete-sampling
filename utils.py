import torch
import matplotlib.pyplot as plt
import numpy as np

# labels is a 1-dimensional tensor
def one_hot(labels, l=10):
    n = labels.shape[0]
    labels = labels.unsqueeze(-1)
    oh = torch.zeros(n, l, device='cuda').scatter_(1, labels, 1)
    return oh

def show_gray_image_grid(imgs, x=2, y=5, size=(20,20), path=None, save=False):
    fig, axs = plt.subplots(x, y, figsize=size)
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(np.squeeze(img), cmap='gray')    
        #ax.imshow(img, cmap='gray')    
        ax.set_axis_off()
        
    if save:
        plt.savefig(path)
    else:
        plt.show() 