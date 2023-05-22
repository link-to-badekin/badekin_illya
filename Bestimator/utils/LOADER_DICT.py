
#Bestimator.utils.LOADER_DICT
import os
import torch
import numpy as np

from skimage.io import imread
import numpy as np
import os
import torch

def read_pair(path, device, model=None, return_feat=False):
    """
        Читает картинку и далает проход для получения logits
        Args:
            path: путь к набору данных с картинками
            model: модель распознавания
        Returns:
            img: Tensor (1, 3, H, W). HxW форма изображения.
            160x160 для FaceNet, 224х224 для VGG, 112x112 is the default.
            feat: Tensor форы (1, model.out_dims).
    """
    img = imread(path).astype(np.float32)
    img = torch.Tensor(img.transpose((2, 0, 1))[None, :]).to(device)
    if not return_feat:
        return img
    feat = model.forward(img).detach().requires_grad_(False)
    return img, feat

class Loader:
    def __init__(self, batch_size, model):
        self.batch_size = batch_size
        self.model = model
        self.device = next(model.parameters()).device
        self.pairs = []
        self.pos = 0
    def __len__(self):
        return len(self.pairs) // self.batch_size
    def __iter__(self):
        return self
    def __next__(self):
        if self.pos < len(self.pairs):
            minibatches_pair = self.pairs[self.pos:self.pos+self.batch_size]
            self.pos += self.batch_size
            xs, ys, ys_feat = [], [], []
            for pair in minibatches_pair:
                path_src, path_dst = pair
                img_src = read_pair(path_src, self.device)
                img_dst, feat_dst = read_pair(path_dst, self.device, self.model, return_feat=True)
                xs.append(img_src)
                ys.append(img_dst)
                ys_feat.append(feat_dst)
            xs = torch.cat(xs)
            ys = torch.cat(ys)
            ys_feat = torch.cat(ys_feat)
            return xs, ys, ys_feat, minibatches_pair
        else:
            raise StopIteration

class LFWLoader(Loader):
    def __init__(self, datadir, goal, batch_size, model):
        super(LFWLoader, self).__init__(batch_size, model)
        with open(os.path.join('config', 'pairs_lfw.txt')) as f:
            lines = f.readlines() 
        suffix = '.jpg'
        self.pairs = []
        for line in lines:
            line = line.strip().split('\t')
            if len(line) == 3 and goal == 'dodging':
                path_src = os.path.join(datadir, line[0], line[0] + '_' + line[1].zfill(4) + suffix)
                path_dst = os.path.join(datadir, line[0], line[0] + '_' + line[2].zfill(4) + suffix)
                self.pairs.append([path_src, path_dst])
            elif len(line) == 4 and goal == 'impersonate':
                path_src = os.path.join(datadir, line[0], line[0] + '_' + line[1].zfill(4) + suffix)
                path_dst = os.path.join(datadir, line[2], line[2] + '_' + line[3].zfill(4) + suffix)
                self.pairs.append([path_src, path_dst])

LOADER_DICT = {
    'lfw': LFWLoader
    #,
}