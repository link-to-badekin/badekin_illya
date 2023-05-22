import kornia
import time
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from Bestimator.utils.data import get_dataloader, store_dataloader
from Bestimator.utils.utils import load_mask, l1_norm, facemask_matrix
from Bestimator.config.config import config_colab
 

def utgt_pgd(mode
    ,model
    , X
    , y
    ,loss_f = nn.CrossEntropyLoss(reduction='none')
    , epsilon=config_colab['pgd']['epsilon']
    , alpha=config_colab['pgd']['alpha']
    , step=config_colab['pgd']['step']
    , univ=False
    , device = None 
):
    if univ:
        delta = torch.zeros_like(X[0], requires_grad=True)
        delta.data = delta.detach() * 2 * epsilon - epsilon
        ub = torch.min(1-X, dim=0)[0]
        lb = torch.max(-X, dim=0)[0]
        delta.data = torch.min(torch.max(delta.detach(), lb), ub)
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        delta.data = delta.detach() * 2 * epsilon - epsilon
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X)

    cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    # обновляем дельту.
    for t in range(step):
        if univ:
            if mode == 'closed':
                loss = torch.min( loss_f( model(X+delta), y))
            else:
                loss = torch.min(1-cos(model(X), model(X+delta)))
        else:
            if mode == 'closed':
                loss = loss_f(model(X+delta), y)
            else:
                loss = torch.sum(1-cos(model(X), model(X+delta)))
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        if univ:
            delta.data = torch.min(torch.max(delta.detach(), lb), ub)
        else:
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X)
        delta.grad.zero_()
                    
    return delta.detach()  

def utgt_occlusion(mode
    , model
    , X
    , y
    , mask
    , loss_f = nn.CrossEntropyLoss(reduction='none')
    , epsilon=config_colab['occlusion']['epsilon']
    , alpha=config_colab['occlusion']['alpha']
    , step=config_colab['occlusion']['step']
    , univ=False
    , device = None
):
    if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if univ:
        delta = torch.ones_like(X[0], requires_grad=True)
    else:
        delta = torch.ones_like(X, requires_grad=True)
    delta = delta.to(device)
    mask = mask.to(device)
    delta.data = delta.detach()*mask*128/255.
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    for t in range(step):
        if univ:
            if mode == 'closed':
                loss = torch.min(loss_f(model(X*(1-mask) + delta), y))
            else:
                loss = torch.min(1-cos(model(X), model(X*(1-mask) + delta)))
        else:
            if mode == 'closed':
                loss = torch.mean(loss_f(model(X*(1-mask) + delta), y), dim=0) 
            else:
                loss = torch.sum(1-cos(model(X), model(X*(1-mask) + delta)))
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(0, epsilon)
        delta.data = delta.detach()*mask
        delta.grad.zero_()
        
    return delta.detach()

def utgt_facemask(mode
    ,model
    , X
    , y
    , mask
    , loss_f = nn.CrossEntropyLoss(reduction='none')
    , epsilon=config_colab['facemask']['epsilon']
    , alpha=config_colab['facemask']['alpha']
    , step=config_colab['facemask']['step']
    , univ=False
    , device = None
):
    mask_left, mask_right, T_left, T_right = facemask_matrix()
    dimension_0 = 1 if univ else X.shape[0]
    temp = torch.zeros(dimension_0, 3, config_colab['facemask']['height'], config_colab['facemask']['width']).to(device)
    # Initialize
    delta = torch.ones_like(temp, requires_grad=True)
    delta.data = delta.detach()*128/255.
    # 2-D
    delta_large = F.interpolate(delta, size=[80, 160])
    delta_large_left = delta_large*mask_left
    delta_large_right = delta_large*mask_right
    # 2-D -> 3-D
    facemask_left = kornia.geometry.transform.warp_perspective(delta_large_left*255., T_left.repeat(dimension_0, 1, 1, 1), dsize=(224, 224))/255.
    facemask_right = kornia.geometry.transform.warp_perspective(delta_large_right*255., T_right.repeat(dimension_0, 1, 1, 1), dsize=(224, 224))/255.
    facemask = facemask_left + facemask_right
    g = torch.zeros_like(delta)
    cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    for t in range(step):
        if univ:
            if mode == 'closed':
                loss = torch.min(loss_f(model(X*(1-mask) + facemask), y))
            else:
                loss = torch.min(1-cos(model(X), model(X*(1-mask) + facemask)))
        else:
            if mode == 'closed':
                loss = loss_f(model(X*(1-mask) + facemask), y)
            else:
                loss = torch.sum(1-cos(model(X), model(X*(1-mask) + facemask)))
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        
        delta_large = F.interpolate(delta, size=[80, 160])
        delta_large_left = delta_large*mask_left
        delta_large_right = delta_large*mask_right
        
        facemask_left = kornia.geometry.transform.warp_perspective(delta_large_left*255., T_left.repeat(dimension_0, 1, 1, 1), dsize=(224, 224))/255.
        facemask_right = kornia.geometry.transform.warp_perspective(delta_large_right*255., T_right.repeat(dimension_0, 1, 1, 1), dsize=(224, 224))/255.
        facemask = facemask_left + facemask_right
        
        delta.grad.zero_()
    facemask.data = torch.min(torch.max(facemask.detach(), -X*(1-mask)), 1-X*(1-mask))
    return facemask.detach()    

class FrameAtack():
    def __init__(self
        ,mask_name = 'occlusion'
        ,utgt_atack = utgt_occlusion
        ,param_atack = config_colab['occlusion']
        ,loss_f = nn.CrossEntropyLoss(reduction='none')
        ,mode = 'closed'
        ,device = None
        ,univ = False
    ):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.mode = mode
        self.utgt_atack = utgt_atack
        self.mask_name = mask_name
        self.univ=univ
        if self.mask_name in config_colab['frame_atack']['masks']:
            self.mask = load_mask(self.mask_name)
            self.mask = self.mask.to(self.device)
        else:
            print(f"Нет такой маски {self.mask_name}, ошибка! ")
            raise ValueError(f"Нет такой маски {self.mask_name}, ошибка! ")
        self.param_atack = param_atack
        self.loss_f = loss_f 
    def attack(self, model, X, y):
        delta = self.utgt_atack (
            mode = self.mode, 
            model = model,
            X = X, 
            y = y,
            loss_f =self.loss_f, 
            epsilon = self.param_atack['epsilon'],
            alpha = self.param_atack['alpha'],
            step = self.param_atack['step'], 
            mask = self.mask, 
            univ=self.univ
        )
        self.mask = self.mask.to(self.device)
        ys = model(X*(1-self.mask)+delta)
        return {'delta' :delta, 'prelog': ys}
