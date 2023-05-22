import torch
import numpy as np

from PIL import Image

def l1_norm(Z):
    return torch.sum(torch.abs(Z.view(Z.shape[0], -1)), 1)[:, None, None, None]

def load_mask(position):
    mask = np.array(Image.open("Bestimator/mask/{}.png".format(position)))/255.
    mask = np.moveaxis(mask, 2, 0)
    mask = torch.from_numpy(mask).float()
    return mask

def facemask_matrix():
    mask_left = torch.zeros(1, 3, 80, 160)
    mask_right = torch.zeros(1, 3, 80, 160)

    mask_left[:, :, :, 0:80] = 1
    mask_right[:, :, :, 80:160] = 1
    mask_left, mask_right = mask_left.cuda(), mask_right.cuda()


    T_left = torch.tensor([[[ 5.0323e-01,  0.0000e+00,  3.2000e+01],
             [-5.9355e-01,  1.0000e+00,  1.1200e+02],
             [-4.4355e-03, -0.0000e+00,  1.0000e+00]]]).cuda()
    T_right = torch.tensor([[[ 5.1556e+00,  0.0000e+00, -1.6356e+02],
             [ 2.0444e+00,  3.4444e+00,  5.8667e+01],
             [ 1.5278e-02,  0.0000e+00,  1.0000e+00]]]).cuda()
    return mask_left, mask_right, T_left, T_right

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
