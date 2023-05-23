#
#Bestimator/feature_extractor/utils.py
import os
import sys

from Bestimator.feature_extractor.MobileFace import MobileFace
from Bestimator.feature_extractor.ResNet import resnet
from Bestimator.feature_extractor.FaceNet import FaceNet
#from Bestimator.feature_extractor.ArcFace import ArcFace
#from Bestimator.feature_extractor.vggface import VGGFace


def get_model(face_model, **kwargs):
    """
        выбриает модель распознавания лица по имени
    """
    img_shape = (112, 112)
    if face_model == 'MobileFace':
        model = MobileFace(**kwargs)
    elif face_model == 'ResNet50':
        model = resnet(depth=50, **kwargs)
    elif face_model == 'FaceNet-VGGFace2':
        model = FaceNet(dataset='vggface2', use_prewhiten=False, **kwargs)
        img_shape = (160, 160)
    elif face_model == 'FaceNet-casia':
        model = FaceNet(dataset='casia-webface', use_prewhiten=False, **kwargs)
        img_shape = (160, 160)
    #elif face_model == 'VGGFace':
    #    model = VGGFace(**kwargs)
    #    img_shape = (224, 224)
    #elif face_model == 'ArcFace':
    #    model = ArcFace(**kwargs)
    else:
        raise NotImplementedError
    return model, img_shape