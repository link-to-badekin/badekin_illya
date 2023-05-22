import sys
import os

from torchvision import datasets, models, transforms
from PIL import Image
from Bestimator.utils.preprocessing_img import preprocessing_with_mtcnn_fix

from Bestimator.config import config_colab
import torch

#from .facial_identiﬁcation_system import Classifier, FacialIdentiﬁcationSystem

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', required=True,
                        help='Root folder where input images are. This folder contains sub-folders for each class.')
    parser.add_argument('--output-folder', required=True, help='Output folder where aligned images will be saved.')
    parser.add_argument('--h-img', required=True, help='size_out')
    parser.add_argument('--w-img', required=True, help='size_out')
    return parser.parse_args()
 
def main():
  args = parse_args()
  preprocessing_with_mtcnn_fix(args.input_folder, args.output_folder, ( int(args.w_img), int(args.h_igm) ))

def default():
  img_size = config_colab['transform_size'] # [(112, 112), (160, 160), (112, 96)]
  ds_name = config_colab['lfw']['name_src']
  path_in = '/content/IMG/lfw-py/lfw_funneled' #os.path.join(config_colab['drive'], ds_name)
  path_out = {s : os.path.join(config_colab['drive'], f"{ds_name}-{s[0]}x{s[1]}") for s in img_size}
  print(path_in)
  for s in img_size:
    print(s, path_out[s])

if __name__ == "__main__":
  main()
