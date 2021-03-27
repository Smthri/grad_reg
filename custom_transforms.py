import torch
from scipy.signal import convolve2d
import numpy as np

def entropy(image):
    p = torch.histc(image, bins=256, min=image.min().item(), max=image.max().item()) / image.shape[0] / image.shape[1]
    p[p == 0.] = 0.0001
    return -torch.dot(p, torch.log2(p)).item()

def QuantizeBatch(batch):
    res = torch.clone(batch)
    
    if len(batch[0].shape) == 3:
        for i, img in enumerate(batch):
            H = 0.
            for j, c in enumerate(img):
                H += entropy(c) / img.shape[0]
                
            k = 64
            if H > 4.:
                k = 128
            if H > 5.:
                k = 256
                
            level = (img.max().item() - img.min().item()) / k
            res[i][img < level] = img.min().item()
            for j in range(1, k-1):
                res[i][(img > j * level) & (img < (j+1) * level)] = j * level + level/2
            res[i][img > (k-1)*level] = img.max().item()
            
            if H > 5.:
                kernel = np.array([[0, 0, 1, 0, 0],
                                   [0, 1, 1, 1, 0],
                                   [1, 1, 1, 1, 1],
                                   [0, 1, 1, 1, 0],
                                   [0, 0, 1, 0, 0]]) / 13.
                for j, c in enumerate(img):
                    res[i, j] = torch.Tensor(convolve2d(c.detach(), kernel, mode='same'))
                
            
        return res
                
    else:
        for i, img in enumerate(batch):
            H = entropy(img)
            
            '''k = 2
            if H > 4.:
                k = 4
            if H > 5.:
                k = 6'''
            
            k = 16
            if H > 4.:
                k = 32
            if H > 5.:
                k = 64
                
            print(H)
                
            level = (img.max().item() - img.min().item()) / k
            res[i][img < level] = img.min().item()
            for j in range(1, k-1):
                res[i][(img > j * level) & (img < (j+1) * level)] = j * level + level/2
            res[i][img > (k-1)*level] = img.max().item()
            
        return res
