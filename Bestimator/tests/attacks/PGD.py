from Bestimator.tests.attacks.restriction_by_norm import ConstrainedMethod
from torch.nn import CosineSimilarity
import torch
import os

__all__ = ['PGD']

class PGD(ConstrainedMethod):
    def __init__(self, model, distance_metric, eps, iters=20, goal = 'dodging', rand_start = True):
        super(PGD, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
        self.rand_start = rand_start
    def batch_attack(self, xs, ys_feat, **kwargs):
        xs_adv = xs.clone().detach().requires_grad_(True)  
        #xs_adv.data = xs_adv.detach() * 2 * self.eps - self.eps
        #delta = delta.detach() * 2 * self.eps - self.eps
        #delta = torch.min(torch.max(delta.detach(), -xs), 1-xs)
        if self.rand_start:
            xs_adv = xs_adv + \
                torch.empty_like(xs_adv).uniform_(-self.eps, self.eps)
            xs_adv = torch.clamp(xs_adv, min=0, max=255).detach().requires_grad_(True)
        for _ in range(self.iters):
            features = self.model.forward(xs_adv)
            loss = self.getLoss(features, ys_feat)
            loss.backward()
            grad = xs_adv.grad
            self.model.zero_grad()
            xs_adv = self.step(xs_adv, 1.5 * self.eps / self.iters, grad, xs, self.eps)
            xs_adv = xs_adv.detach().requires_grad_(True)
        return torch.clamp(xs_adv, min=0, max=255)