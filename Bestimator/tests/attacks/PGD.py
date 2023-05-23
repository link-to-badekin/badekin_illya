from Bestimator.tests.attacks.restriction_by_norm import ConstrainedMethod
from torch.nn import CosineSimilarity
import torch

__all__ = ['PGD']

class PGD(ConstrainedMethod):
    def __init__(self, model, distance_metric, eps, iters=20, goal = 'dodging'):
        super(PGD, self).__init__(model, goal, distance_metric, eps)
        self.iters = iters
    def batch_attack(self, xs, ys_feat, **kwargs):
        xs_adv = xs.clone().detach().requires_grad_(True)
        xs_adv.data = xs_adv.detach() * 2 * epsilon - epsilon
        xs_adv.data = torch.min(torch.max(delta.detach(), -xs), 1-xs)
        for _ in range(self.iters):
            features = self.model.forward(xs_adv)
            loss = self.getLoss(features, ys_feat)
            loss.backward()
            grad = xs_adv.grad
            self.model.zero_grad()
            xs_adv = self.step(xs_adv, 1.5 * self.eps / self.iters, grad, xs, self.eps)
            xs_adv = xs_adv.detach().requires_grad_(True)
        return xs_adv