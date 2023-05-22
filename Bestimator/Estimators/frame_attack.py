import kornia
import time
import copy
import os
import torch

import torch.nn as nn
import torch.nn.functional as F

import os
from Bestimator.data import get_dataloader, store_dataloader
from Bestimator.utils import load_mask, l1_norm, facemask_matrix
#from Bestimator.attack. import utgt_pgd, utgt_occlusion, utgt_facemask
from Bestimator.tests.attacks.frame_attacks import utgt_pgd, utgt_occlusion, utgt_facemask
from Bestimator.tests.attacks.frame_attacks import FrameAtack
from Bestimator.config.config import config_colab


def eval_robustness_frame_atack_closed(face_id_sys
	, atack
	, output_datapath 
	, batch_size = config_colab['batch_size']
	, device = None
	, performance_metric = None
): 
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if performance_metric is None:
		#если нет специальной, то  accuracy
		p_m = lambda pm_preds, pm_labels, pm_loader: torch.sum(pm_preds == pm_labels).double()/len(pm_loader.dataset)
		performance_metric = {'name': 'Accuracy', 'call': p_m}

	model_fc = face_id_sys.model_fc
	model_fc.eval()

	test_loader = get_dataloader(face_id_sys.test_gallery, batch_size, face_id_sys.img_size)
	labels = sorted(os.listdir(face_id_sys.test_gallery))
	running_corrects = 0.0	
	
	delta = 0
	confidence_res = []
	for batch_idx, (X, y) in enumerate(test_loader):
		X, y = X.to(device), y.to(device)
		res_atack = atack.attack(X = X, y = y, model = model_fc)
		delta = res_atack['delta']
		ys = res_atack['prelog']
		conf, yp = torch.max(ys, 1)
		confidence_res.append(conf)
		running_corrects += performance_metric['call'](yp, y.data, test_loader)
		#копим состязательные примеры
		print(f"idx {batch_idx}")
		if batch_idx == 0:
				X_adv = X*(1-atack.mask)+delta
				y_adv = y 
		else:
				X_adv = torch.cat((X_adv, X*(1-atack.mask)+delta), 0)
				y_adv = torch.cat((y_adv, y), 0)		
	adv_dataset = torch.utils.data.TensorDataset(X_adv.cpu().detach(), y_adv.cpu().detach())
	adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, shuffle=False)

	store_dataloader(adv_loader, output_datapath, labels)
	try:
		print("{} on adversarial examples to be stored in {}: {}".format(
		performance_metric['name'], output_datapath, running_corrects))
		average_confidence = torch.sum(torch.tensor (confidence_res) )/len(confidence_res)
		print(f"average confidence {average_confidence}")
	except Exception as e:
		print(e)
	return adv_loader