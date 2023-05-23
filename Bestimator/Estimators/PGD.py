import numpy as np
import torch
import os
import argparse

from Bestimator.feature_extractor.utils import get_model
from Bestimator.Estimators.run import run_test
from Bestimator.utils.LOADER_DICT import LOADER_DICT
from Bestimator.tests.attacks.PGD import  PGD

parser = argparse.ArgumentParser()
parser.add_argument('--device', help='device id', type=str, default='cuda')
parser.add_argument('--dataset', help='dataset', type=str, default='lfw', choices=['lfw'])
parser.add_argument('--model', help='White-box model', type=str, default='MobileFace')
parser.add_argument('--goal', help='dodging/impersonate', type=str, default='impersonate', choices=['dodging', 'impersonate'])
parser.add_argument('--eps', help='epsilon', type=float, default=16)
parser.add_argument('--iters', help='attack iteration', type=int, default=20)
parser.add_argument('--seed', help='random seed', type=int, default=1234)
parser.add_argument('--batch_size', help='batch_size', type=int, default=20)
parser.add_argument('--distance', help='l2 or linf', type=str, default='linf', choices=['linf', 'l2'])
parser.add_argument('--output', help='output dir', type=str, default='output/expdemo')
parser.add_argument('--datapath', help='input dir', type=str, default='data')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def pgd_estimator(
    datapath  = 'data' 
    ,device = 'cuda'
    ,dataset = 'lfw'
    ,model = 'MobileFace'
    ,goal = 'dodging'
    ,eps = 4
    ,iters = 20
    ,seed = 1234
    ,batch_size = 20
    ,distance = 'linf'
    ,output = 'result'
):
    """"Обвертка для PGD"""
    model, img_shape = get_model(model, device = device)
    attacker = PGD(
        model=model,
        goal=goal, 
        distance_metric=distance, 
        eps=eps,
        iters=iters
    )
    datapath = os.path.join( datapath, '{}-{}x{}'.format(dataset, img_shape[0], img_shape[1]))
    print(f"Полный путь к набору данных: {datapath}")
    loader = LOADER_DICT[dataset](datapath, goal, batch_size, model)
    run_test(loader, attacker, output)

def main():
    print(f"Запускаю PGD")
    pgd_estimator(
    datapath  = args.datapath 
    ,device = args.device
    ,dataset =args.dataset
    ,model = args.model
    ,goal = args.goal
    ,eps = args.eps
    ,iters = args.iters
    ,seed = args.seed
    ,batch_size = args.batch_size
    ,distance = args.distance
    ,output = args.output
    )

if __name__ == '__main__':
    main()
