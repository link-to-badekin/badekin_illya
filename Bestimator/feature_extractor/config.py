threshold_lfw = {
	'FaceNet-VGGFace2':{'cos': 0.42074376344680786, 'cos_acc': 0.9923333333333333},
	'MobileFace':{'cos': 0.21116749942302704, 'cos_acc': 0.9945},
	'FaceNet-casia':{'cos': 0.4289606213569641, 'cos_acc': 0.981},
    'ArcFace':{'cos_acc': 0.995, 'cos': 0.28402677178382874},
	'ResNet50':{'cos': 0.19116485118865967, 'cos_acc': 0.9971666666666666},
	'ResNet50-casia':{'cos': 0.1854616403579712, 'cos_acc': 0.986},
}

THRESHOLD_DICT = {
    'lfw': threshold_lfw
    #,
}