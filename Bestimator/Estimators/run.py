import os
from skimage.io import imread, imsave
import numpy as np
import torch
from tqdm import tqdm

from Bestimator.feature_extractor.config import threshold_lfw

def cosdistance(x, y, offset=1e-5):
    """
        Нормируем, покоорд перемнож.
        Args:
            x: tensor (1, n)
            y: tensor (1, n)
    """
    x = x / torch.sqrt(torch.sum(x**2)) + offset
    y = y / torch.sqrt(torch.sum(y**2)) + offset
    return torch.sum(x * y)

def L2distance(x, y):
    """
        L2 distance.
        Args:
            x: tensor (1, n)
            y: tensor (1, n)
    """
    return torch.sqrt(torch.sum((x - y)**2))

def save_images(image, original_image, filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image = np.clip(image, 0, 255).astype(np.uint8)
    imsave(os.path.join(output_dir, filename), image.astype(np.uint8))

def binsearch_basic(
	xs,
	ys,
	ys_feat,
	pairs,
	eps,
	method,
	threshold,
	steps=0,
	bin_steps=0,
	*args, 
	**kwargs
):
    """
        Бинарный поиск для оптимального параметра eps для FGSM   
        return:
        	xs_results: tensor (batch, 3, H, W). adversarial examples for xs.
    """
    batch_size = xs.size(0)
    lo = torch.zeros(batch_size, 1, 1, 1).cuda()
    hi = lo + eps
    eps_tensor = torch.zeros(batch_size, 1, 1, 1).cuda() + eps
    exp = 2 ** steps
    xs_results = xs.clone().detach()
    goal = kwargs['goal']
    model = kwargs['model']
    found = torch.zeros(batch_size).bool().cuda()
    # возможный EPS по линейному поиску
    for _ in range(exp):
        magnitude = (1.0 - float(_) / exp) * eps_tensor
        kwargs['eps'] = magnitude
        Attacker = method(**kwargs)
        xs_adv = Attacker.batch_attack(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
        ys_adv = model.forward(xs_adv)
        similarities = torch.sum(ys_adv * ys_feat, dim=1)
        if goal == 'dodging':
            succ_ = threshold - similarities > 0
        else:
            succ_ = similarities - threshold > 0
        xs_results[succ_] = xs_adv[succ_]
        hi[succ_] = (1.0 - float(_) / exp) * eps
        found[succ_] = True
    lo = hi - float(eps) / exp
    # по бинарному
    for i in range(bin_steps):
        mi = (lo + hi) / 2
        kwargs['eps'] = mi
        Attacker = method(**kwargs)
        xs_adv = Attacker.batch_attack(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
        ys_adv = model.forward(xs_adv)
        similarities = torch.sum(ys_adv * ys_feat, dim=1)
        if goal == 'dodging':
            succ_ = threshold - similarities > 0
        else:
            succ_ = similarities - threshold > 0
        hi[succ_] = mi[succ_]
        lo[~succ_] = mi[~succ_]
        xs_results[succ_] = xs_adv[succ_]
        found[succ_] = True
    y = model.forward(xs)
    similarities = torch.sum(y * ys_feat, dim=1)
    if goal == 'dodging':
        succ_ = threshold - similarities > 0
    else:
        succ_ = similarities - threshold > 0
    xs_results[succ_] = xs[succ_]
    found[succ_] = True
    return xs_results, found

def binsearch_alpha(
	xs
	, ys
	, ys_feat
	, pairs
	, eps
	, method
	, threshold
	, steps=0
	, bin_steps=0
	, *args
	, **kwargs
):
    """
        xs_results: tensor (batch, 3, H, W). adversarial examples for xs.
    """
    batch_size = xs.size(0)
    lo = torch.zeros(batch_size, 1, 1, 1).cuda()
    hi = lo + eps
    xs_results = xs.clone().detach()
    goal = kwargs['goal']
    model = kwargs['model']
    found = torch.zeros(batch_size).bool().cuda()
    for _ in range(steps):
        kwargs['eps'] = hi
        Attacker = method(**kwargs)
        xs_adv = Attacker.batch_attack(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
        ys_adv = model.forward(xs_adv)
        similarities = torch.sum(ys_adv * ys_feat, dim=1)
        if goal == 'dodging':
            succ_ = threshold - similarities > 0
        else:
            succ_ = similarities - threshold > 0
        cond = ~found & succ_
        xs_results[cond] = xs_adv[cond]
        found[cond] = True
        not_found = ~found
        lo[not_found] = hi[not_found]
        hi[not_found] *= 2
        if found.all():
            break
    for i in range(bin_steps):
        mi = (lo + hi) / 2
        kwargs['eps'] = mi
        Attacker = method(**kwargs)
        xs_adv = Attacker.batch_attack(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
        ys_adv = model.forward(xs_adv)
        similarities = torch.sum(ys_adv * ys_feat, dim=1)
        if goal == 'dodging':
            succ_ = threshold - similarities > 0
        else:
            succ_ = similarities - threshold > 0
        hi[succ_] = mi[succ_]
        lo[~succ_] = mi[~succ_]
        xs_results[succ_] = xs_adv[succ_]
        found[succ_] = True
    y = model.forward(xs)
    similarities = torch.sum(y * ys_feat, dim=1)
    if goal == 'dodging':
        succ_ = threshold - similarities > 0
    else:
        succ_ = similarities - threshold > 0
    xs_results[succ_] = xs[succ_]
    found[succ_] = True
    return xs_results, found

def run_test(loader, Attacker, output):
    os.makedirs(output, exist_ok=True)
    cnt = 0
    outputs = []
    for xs, ys, ys_feat, pairs in tqdm(loader, total=len(loader)):
        x_adv = Attacker.batch_attack(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
        for i in range(len(pairs)):
            img = x_adv[i].detach().cpu().numpy().transpose((1, 2, 0))
            cnt += 1
            npy_path = os.path.join(output, str(cnt) + '.npy')

            outputs.append([npy_path, pairs[i][0], pairs[i][1]])
            np.save(npy_path, img)
            original_image = xs[i].cpu().numpy().transpose((1, 2, 0))
            save_images(img, original_image, str(cnt) + '.png', output)
    with open(os.path.join(output, 'annotation.txt'), 'w') as f:
        for pair in outputs:
            f.write('{} {} {}\n'.format(pair[0], pair[1], pair[2]))


def run_white(loader, Attacker, model, output, distance, log):
    os.makedirs('log', exist_ok=True)
    cnt = 0
    scores = []
    dists = []
    success = []
    advs = []
    imgs = []
    for xs, ys, ys_feat, pairs in tqdm(loader, total=len(loader)):
        x_adv, found = Attacker(xs=xs, ys=ys, ys_feat=ys_feat, pairs=pairs)
        y_adv = model.forward(x_adv)
        s = torch.sum(y_adv * ys_feat, dim=1)
        for i in range(len(pairs)):
            img = x_adv[i].detach().cpu().numpy().transpose((1, 2, 0))
            x = xs[i].detach().cpu().numpy().transpose((1, 2, 0))
            scores.append(s[i].item())
            success.append(int(found[i].item()))
            cnt += 1
            advs.append(str(cnt) + '.npy')
            npy_path = os.path.join(output, str(cnt) + '.npy')
            np.save(npy_path, img)
            if distance == 'l2':
                dist = np.linalg.norm(img - x) / np.sqrt(img.reshape(-1).shape[0])
            else:
                dist = np.max(np.abs(img - x))
            dists.append(dist)
            imgs.append(pairs[i][1])
            original_image = xs[i].cpu().numpy().transpose((1, 2, 0))
            save_images(img, original_image, str(cnt) + '.png', output)

    with open(os.path.join('log', log), 'w') as f:
        f.write('adv_img,tar_img,score,dist,success\n')
        for adv, img, score, d, s in zip(advs, imgs, scores, dists, success):
            f.write('{},{},{},{},{}\n'.format(adv, img, score, d, s))


