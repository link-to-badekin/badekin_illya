import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchfile


from typing import Union, Tuple

#from config import *


class vgg_face(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.size_out = 4096
        #self.fc7 = nn.Linear(1024, 1024)
        #self.fc8 = nn.Linear(1024, out_classes)

    def load_weights(self, path="Bestimator/model/pretrained_vggface.t7"):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                    # only load convolutional layers, learn fc layers by our own examples
                else:
                    #загружаем последний слой для представления embedding  
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                    break



    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x
        # 
        #x = F.dropout(x, 0.5, self.training)
        #x = F.relu(self.fc7(x))
        #x = F.dropout(x, 0.5, self.training)
        #return self.fc8(x)

class vgg_face_open(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

    def load_weights(self, path="Bestimator/model/pretrained_vggface.t7"):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc8(x)

class normalizeLayer(torch.nn.Module):
    def __init__(self, means = None, sds = None, device = None):
        super(normalizeLayer, self).__init__()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.means = torch.tensor(means).cuda()
        self.means = torch.tensor(means).to(device)
        #self.sds = torch.tensor(sds).cuda()
        self.sds = torch.tensor(sds).to(device)
    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
    
def get_normalize_layer(means, stds):
    return normalizeLayer(means, stds)

class rgb2bgrLayer(torch.nn.Module):
    def __init__(self):
        super(rgb2bgrLayer, self).__init__()

    def forward(self, input: torch.tensor):
        return input[:,[2,1,0],:,:]

def get_rgb2bgr_layer():
    return rgb2bgrLayer()

def resize(input: torch.Tensor, size: Union[int, Tuple[int, int]], interpolation: str = 'bilinear') -> torch.Tensor:
    """Resize the input torch.Tensor to the given size.
    See :class:`~kornia.Resize` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    new_size: Tuple[int, int]

    if isinstance(size, int):
        w, h = input.shape[-2:]
        if (w <= h and w == size) or (h <= w and h == size):
            return input
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        new_size = (ow, oh)
    else:
        new_size = size

    return torch.nn.functional.interpolate(input, size=new_size, mode=interpolation)
    
class Resize(nn.Module):
    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: str = 'bilinear') -> None:
        super(Resize, self).__init__()
        self.size: Union[int, Tuple[int, int]] = size
        self.interpolation: str = interpolation

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return resize(input, self.size, self.interpolation)

def get_resize_layer(size):
    return Resize(size)

def get_pretrained_vggface(mode, input_size = 224, path = None ,device = None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
    dataset_mean, dataset_std = [0.367035294117647,0.41083294117647057,0.5066129411764705], [1/255., 1/255., 1/255.]
    #авторесайз входа
    resize_layer = get_resize_layer(input_size)
    #нормализация
    normalize_layer = get_normalize_layer(dataset_mean, dataset_std)
    #цвета
    rgb2bgr_layer = get_rgb2bgr_layer()
    if mode == 'closed':
        model = vgg_face()
    else:
        model = vgg_face_open()
    model.load_weights(path)
    model =  model.to(device)
    model = torch.nn.Sequential(resize_layer.to(device), rgb2bgr_layer, normalize_layer, model)
    #model = torch.nn.Sequential(resize_layer.cuda(), rgb2bgr_layer, normalize_layer, model.cuda())
    return model