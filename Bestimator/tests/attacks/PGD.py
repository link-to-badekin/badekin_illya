from Bestimator.tests.attacks.restriction_by_norm import ConstrainedMethod
from torch.nn import CosineSimilarity
import torch
import os

__all__ = ['PGD']

class PGD(ConstrainedMethod):
    def __init__(self, model, distance_metric, eps, iters=20, goal = 'dodging'):
        super(PGD, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
    def batch_attack(self, xs, ys_feat, **kwargs):
        delta = torch.zeros_like(xs, requires_grad=True) 
        #xs.clone().detach().requires_grad_(True)
        #xs_adv.data = xs_adv.detach() * 2 * self.eps - self.eps
        delta = delta.detach() * 2 * self.eps - self.eps
        delta = torch.min(torch.max(delta.detach(), -xs), 1-xs)
        for _ in range(self.iters):
            xs_adv = xs+delta
            features = self.model.forward(xs_adv)
            loss = self.getLoss(features, ys_feat)
            loss.backward()
            grad = xs_adv.grad
            self.model.zero_grad()
            delta = self.step(delta, 1.5 * self.eps / self.iters, grad, 0, self.eps)
            delta = delta.detach().requires_grad_(True)
        xs_adv = xs+delta
        return torch.clamp(xs_adv, min=0, max=255)