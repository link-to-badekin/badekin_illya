import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
from Bestimator.config.config import config_colab

class Classifier(nn.Module):
	def __init__(self, feature = None, out_f = None, F_act = None, logits = False):
		"""
		feature - размер вектора фичей
		out_f - количетсво классов в галерее
		"""
		super().__init__()
		if feature is None:
			raise ValueError("Нет размерности признакового пространства")
		if out_f is None:
			raise ValueError("Нет размерности выходных классов")	
		if F_act is None:
			self.F_act = F.relu
		self.logits = logits
		self.fc7 = nn.Linear(feature, feature)
		self.fc8 = nn.Linear(feature, out_f)
	def forward(self, x):
		"""logits """
		x = self.F_act(x) 
		x = F.dropout(x, 0.5, self.training)
		x = self.F_act(self.fc7(x))
		x = F.dropout(x, 0.5, self.training)
		if self.logits:
			return self.fc8(x)
		#возвращаем вероятности
		return F.softmax(self.fc8(x), dim=1) 

def load_weights(model, path):
	print(f"Загружаю модель из {path}")
	return model.load_state_dict(torch.load(path))

def train_classifier(
	classifier,
	feature_extractor,
	dataloaders, 
	criterion = None, 
	optimizer = None,  
	num_epochs=25, 
	scheduler = None,
	performance_metric = None,
	device = None
):
	#блок штук по умолчанию
	if criterion is None:
		criterion = nn.CrossEntropyLoss()
	if optimizer is None:
		optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
	if scheduler is  None:
		scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if performance_metric is None:
		#если нет специальной, то  accuracy
		p_m = lambda pm_preds, pm_labels, pm_loader: torch.sum(pm_preds == pm_labels).double()/len(pm_loader.dataset)
		performance_metric = {'name': 'Accuracy', 'call': p_m} 
	since = time.time()
	best_model_wts = copy.deepcopy(classifier.state_dict())
	best_acc = 0.0
	feature_extractor.eval()
	feature_extractor.requires_grad = False
	model = classifier
	for epoch in range(num_epochs):
		# Начинаем
		print(f'Epoch {epoch}/{num_epochs - 1}')
		print('+' * 10)
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			running_corrects = 0
			
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)
				
				optimizer.zero_grad()
				#print(inputs, labels)
				if phase == 'train':
					model.requires_grad = True
				else:
					model.requires_grad = False
				feature = feature_extractor(inputs)
				outputs = model(feature)
				if model.logits:
					outputs = F.softmax(outputs, dim=1)
				_, preds = torch.max(outputs, 1)
				#print(outputs)
				loss = criterion(outputs, labels)
				#шаг оптимизатора
				if phase == 'train':
					loss.backward()
					optimizer.step()
				running_loss += loss.item() * inputs.size(0)
				running_corrects += performance_metric['call'](preds, labels.data, dataloaders[phase])
			#шаг планера
			if phase == 'train':
				scheduler.step()

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects 
			#running_corrects.double() / dataset_sizes[phase]
			print('{} Loss: {:.4f} {}: {:.4f}'.format(phase, epoch_loss, performance_metric['name'], epoch_acc))
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
	time_elapsed = time.time() - since
	print('#'*10)
	print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
	print(f'Best val Acc: {best_acc:4f}')
	model.load_state_dict(best_model_wts)
	return model

def train_classifier_emb(
	classifier,
	dataloaders, 
	criterion = None, 
	optimizer = None,  
	num_epochs=25, 
	scheduler = None,
	performance_metric = None,
	device = None
):
	"""
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD( self.classifier.parameters(), lr=0.001, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
	performance_metric:
		p_m = lambda pm_preds, pm_labels, pm_loader: torch.sum(_preds == labels)/len(pm_loader.dataset)
		performance_metric = {'name': 'Accuracy', 'call': p_m}
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	dataloaders - embeddings	
	"""
	#блок штук по умолчанию
	if criterion is None:
		criterion = nn.CrossEntropyLoss()
	if optimizer is None:
		optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
	if scheduler is  None:
		scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if performance_metric is None:
		#если нет специальной, то  accuracy
		p_m = lambda pm_preds, pm_labels, pm_loader: torch.sum(_preds == labels).double()/len(pm_loader.dataset)
		performance_metric = {'name': 'Accuracy', 'call': p_m} 
	since = time.time()
	best_model_wts = copy.deepcopy(classifier.state_dict())
	best_acc = 0.0
	model = classifier
	for epoch in range(num_epochs):
		# Начинаем
		print(f'Epoch {epoch}/{num_epochs - 1}')
		print('+' * 10)
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			running_corrects = 0
			
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()
				#print(inputs, labels)
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs) 
					# получаем вероятности (до этого были логиты)
					if model.logits:
						outputs = F.softmax(outputs, dim=1)
					_, preds = torch.max(outputs, 1)
					#print("It is outputs")
					print(outputs)
					loss = criterion(outputs, labels)
					#шаг оптимизатора
					if phase == 'train':
						loss.backward()
						optimizer.step()
				running_loss += loss.item() * inputs.size(0)
				running_corrects += performance_metric['call'](preds, labels.data, dataloaders[phase])
			#шаг планера
			if phase == 'train':
				scheduler.step()
			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects 
			print('{} Loss: {:.4f} {}: {:.4f}'.format(phase, epoch_loss, performance_metric['name'], epoch_acc))
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
	time_elapsed = time.time() - since
	print('#'*10)
	print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
	print(f'Best val Acc: {best_acc:4f}')
	model.load_state_dict(best_model_wts)
	return model

def make_dataloader_from_embeddings(
	extractor = None,
	image_datasets = None, 
	batch_size = config_colab['batch_size'], 
	shuffle = True,
	device = None
):
	"""
	extractor - модель 
	image_datasets - дс фолдер
	return None, если что-то не так
	"""
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if extractor is None or image_datasets is None:
		return None 
	#извлекаем признаки
	dataloader = torch.utils.data.DataLoader(image_datasets, batch_size = batch_size,
										shuffle=False, num_workers=4)
	all_labels = None
	all_outputs	= None
	#делаем дс признаков
	for inputs, labels in dataloader:
		inputs = inputs.to(device)
		
		if all_outputs is None:
			all_outputs = extractor(inputs)
			all_labels = labels
			continue
		outputs =  extractor(inputs) 	
		all_outputs = torch.cat((all_outputs, outputs), dim = 0)
		all_labels = torch.cat( (all_labels, labels),  dim = 0)

	#outputs = extractor(image_datasets)
	print(f"Создал тензор признаков")
	print(all_outputs.shape)
	print(torch.sum(all_labels == image_datasets.classes))
	dataset = torch.utils.data.TensorDataset(all_outputs , all_labels)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=shuffle)
	return dataloader

class FacialIdentiﬁcationSystem():
	"""Facial Identiﬁcation System"""
	def __init__(self
		, extractor = None
		, classifier = None
		, detector = None
		, classify = True
		, device = None
		, img_size = (224,224)
	):
		#super().__init__()
		if device is None:
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		if classifier is None:
			print(f"Нет модуля классификации")
		self.classifier = classifier

		if extractor is None:
			print(f"Нет модуля извлечения признаков")
		self.extractor = extractor

		if detector is None:
			self.extractor = self.extractor
			self.model_fc = torch.nn.Sequential(self.extractor, self.classifier)
		else:
			self.detector = detector
			self.extractor = torch.nn.Sequential(self.detector, self.extractor)
			self.model_fc = torch.nn.Sequential(self.detector, self.extractor, self.classifier)
		
		self.train_gallery = None
		self.test_gallery = None
		self.dataloaders = None
		self.classify = classify
		self.img_size = img_size 
		print("Собрана модель биометрической идентификации пользователей по лицу")
	def forward(self, x):
		if self.classify:
			x = self.model_fc(x)
		else:
			x = self.extractor(x)
		return x
	def get_featurs(self, x):
		return self.extractor(x)
	def make_dataloader(self, batch_size = 50):
		if self.train_gallery is None or self.test_gallery is None:
			print(f"Не задана галерея: train = {self.train_gallery} | test = {self.test_gallery}")
		
		pathes = {'train': self.train_gallery ,'val': self.test_gallery}
		
		data_transforms =  transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
		])
		image_datasets = {x: datasets.ImageFolder(pathes[x], data_transforms)
				  for x in ['train', 'val']}

		self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
										shuffle=True, num_workers=4)
			for x in ['train', 'val']}
		return self.dataloaders

	def set_train_gallery(self, path):
		self.train_gallery = path
	def set_test_gallery(self, path):
		self.test_gallery = path
	def eval_folder(self
		,data_path = None
		,batch_size = config_colab['batch_size']
		,performance_metric = None
	):
		"""
		Evaluate closed-set face recognition on data stored in test_datapath.
		Metric: prediction accuracy
		""" 
		self.model_fc.eval()
		#self.model
		if performance_metric is None:
		#если нет специальной, то  accuracy
			p_m = lambda pm_preds, pm_labels, pm_loader: torch.sum(pm_preds == pm_labels).double()/len(pm_loader.dataset)
			performance_metric = {'name': 'Accuracy', 'call': p_m}

		if data_path == None:
			data_path = self.test_gallery
		test_loader = get_dataloader(data_path, config_colab['batch_size'])

		running_corrects = 0.0
		confidence_total = 0.0	

		with torch.no_grad():
			for X, y in test_loader:
				X, y = X.to(self.device), y.to(self.device)

				# Forward
				outputs = self.model_fc(X)
				if self.classifier.logits:
					outputs = F.softmax(outputs, dim=1)
					
				confidence, yp = torch.max(outputs, 1)
				confidence_total += torch.sum(confidence) 
				running_corrects += performance_metric['call'](yp, y.data, test_loader)
		
		#print('{} Loss: {:.4f} {}: {:.4f}'.format(phase, epoch_loss, performance_metric['name'], epoch_acc))		
		print("{} на данных {}: {}".format(performance_metric['name'], data_path, running_corrects))
		print(f" Средняя уверенность в ответах = {confidence_total/len(test_loader.dataset)}")
	
def make_classifier( 
	feature_extractor
	,img_size = (224,224)
	,path_train = os.path.join(config_colab['drive'], config_colab['VGG2TEST']['train_ds'])
 	,path_test = os.path.join(config_colab['drive'], config_colab['VGG2TEST']['test_ds'])
 	,save_path = config_colab['drive']
 	,model_name = 'fc'
 	,batch_size = config_colab['batch_size']
 	,out_f = 10
 	,criterion = None
 	,data_transforms = None
 	,num_epochs = 24
 	,device = None
 	,optimizer = None  
	,scheduler = None,
	performance_metric = None

):
	"""Создает, обучает и сохраняет нужный классификатор"""
	if data_transforms is None:	
		data_transforms =  transforms.Compose([
			transforms.Resize(img_size),
			transforms.ToTensor(),
		])
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	img_train_fold = datasets.ImageFolder(path_train, data_transforms)
	train_loader = torch.utils.data.DataLoader(img_train_fold, batch_size=batch_size, shuffle=True)
	img_test_fold = datasets.ImageFolder(path_test, data_transforms)
	test_loader = torch.utils.data.DataLoader(img_test_fold, batch_size=batch_size, shuffle=False)
	fc = Classifier(feature_extractor[-1].size_out, out_f)
	dataloaders = {'train': train_loader, 'val': test_loader }
	feature_extractor = feature_extractor.to(device)
	fc = fc.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(fc.parameters(), lr=0.001, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5)
	
	fc = train_classifier(
		classifier = fc,
		feature_extractor = feature_extractor, 
		criterion = criterion, 
		optimizer = optimizer,  
		num_epochs= num_epochs, 
		scheduler = scheduler,
		performance_metric = performance_metric,
		device = device,
		dataloaders = dataloaders
	)
	ds_name = path_train.split('/')[-1]
	name_model = f"{model_name}_ebm{feature_extractor[-1].size_out}_c{out_f}_{ds_name}_b{batch_size}_ep{num_epochs}.pt" 
	save_path = os.path.join(save_path, name_model)
	print(f"safe {save_path}")
	torch.save(fc.state_dict(), save_path)
	return fc