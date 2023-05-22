import os

from torchvision import datasets, transforms
from facenet_pytorch.models.mtcnn import MTCNN
from PIL import Image

import torch

def create_dirs(root_dir, classes):
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
    for clazz in classes:
        path = root_dir + os.path.sep + clazz
        if not os.path.isdir(path):
            os.mkdir(path)

 
def crop_center(pil_img, crop_width: int, crop_height: int) -> Image:
    """
    Функция для обрезки изображения по центру.
    """
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def preprocessing_with_mtcnn(src_path, out_path, size_img, device = None, post_process = False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_dim = size_img[0] if size_img[0] > size_img[1] else size_img[1]

    images = datasets.ImageFolder(root=src_path)
    trans = transforms.Compose([
        transforms.Resize(min_dim*1.2)
    ])
    images = datasets.ImageFolder(root=src_path)
    #получаем список папок и файлов
    images.idx_to_class = {v: k for k, v in images.class_to_idx.items()}
    create_dirs(out_path, images.classes)
    mtcnn = MTCNN(post_process = post_process, device = device, image_size = max_dim)

    for idx, (path, y) in enumerate(images.imgs):
        print("Выравниваю {} {}/{} ".format(path, idx + 1, len(images)), end='')
        aligned_path = out_path + os.path.sep + images.idx_to_class[y] + os.path.sep + os.path.basename(path)
        if not os.path.exists(aligned_path):
            img = mtcnn(img=(Image.open(path).convert('RGB')), save_path=aligned_path)
            print("Лицо не найдено" if img is None else '')
            print(aligned_path)
        else:
            print('Уже готово')



def preprocessing_with_mtcnn_fix(src_path, out_path, size_img, device = None, post_process = False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_dim = size_img[0] if size_img[0] > size_img[1] else size_img[1]
    max_dim = size_img[0] if size_img[0] < size_img[1] else size_img[1]

    images = datasets.ImageFolder(root=src_path)
    #получаем список папок и файлов
    images.idx_to_class = {v: k for k, v in images.class_to_idx.items()}
    #создаем папки
    create_dirs(out_path, images.classes)
    #без специальной обработки от MTCNN 
    mtcnn = MTCNN( post_process = post_process, device = device, image_size = max_dim)

    for idx, (path, y) in enumerate(images.imgs):
        print("Выравниваю {} {}/{} ".format(path, idx + 1, len(images)), end='')
        aligned_path = out_path + os.path.sep + images.idx_to_class[y] + os.path.sep + os.path.basename(path)
        if not os.path.exists(aligned_path):
            img = mtcnn(img=(Image.open(path).convert('RGB')), save_path=aligned_path)
            print("Лицо не найдено" if img is None else '')
            print(aligned_path)
        else:
            print('Уже готово')

def extract_faces(src_path, out_path, device = None, post_process = False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    images = datasets.ImageFolder(root=src_path)
    #получаем список папок и файлов
    images.idx_to_class = {v: k for k, v in images.class_to_idx.items()}
    #создаем папки
    create_dirs(out_path, images.classes)
    #без специальной обработки от MTCNN 
    mtcnn = MTCNN( post_process = post_process, device = device)

    for idx, (path, y) in enumerate(images.imgs):
        print("Выравниваю {} {}/{} ".format(path, idx + 1, len(images)), end='')
        aligned_path = out_path + os.path.sep + images.idx_to_class[y] + os.path.sep + os.path.basename(path)
        if not os.path.exists(aligned_path):
            img = mtcnn(img=(Image.open(path).convert('RGB')), save_path=aligned_path)
            print("Лицо не найдено" if img is None else '')
            print(aligned_path)
        else:
            print('Уже готово')