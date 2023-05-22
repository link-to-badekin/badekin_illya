import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

def get_dataloader(data_path, batch_size = 50, img_size = (224,224) ):
    data_transforms =  transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
    ])
    
    image_datasets = datasets.ImageFolder(data_path, data_transforms) 
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=False)
        
    dataset_size = len(image_datasets) 
    print("Размер набора данных: ", dataset_size)
    return dataloader

def plot_sample(X, M, N):
    """
    Args:
        X (torch.Tensor): the image batch to plot.
        M (int): rows of image to plot.
        N (int): columns of image to plot.
    """
    #%matplotlib inline

    print("Show the images...")
    f,ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N*3, M*3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(X[i*N+j].cpu().detach().numpy().transpose((1, 2, 0)))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    #plt.show();

def store_dataloader(data_loader, data_path, labels):
    print('Начинаю сохранение')
    if not os.path.exists(data_path):
        os.system('mkdir {}'.format(data_path))
        # Create subfolders for each identity
        print('создаю дирректории')
        for label in labels:
            os.system('mkdir {}/{}'.format(data_path, label))

    # Then store images returned by data_loader.
    i = 0
    print('Сохраняю')
    for X, y in data_loader:
        print(X.shape)
        print(y) 
        X = X.cpu().detach().numpy()
        X = X.transpose((1, 2, 0))
        X = np.clip(X, 0, 1)*255
        X = Image.fromarray(X.astype('uint8'))
        
        X.save('{}/{}/{}.png'.format(data_path, labels[y.item()], i))
        i += 1