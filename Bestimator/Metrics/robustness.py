#from sklearn.metrics import roc_curve
import torch
import numpy as np



def l1_norm(Z):
    """
    Compute the l1 norm of a given tensor. This is used to compute the momentum when solving optimization problems.
    Args:
        Z (torch.Tensor): input image with size [batch, 3, 224, 224].
    Returns:
        l1 norm of Z (torch.Tensor): reshaped l1 norm with size [batch, 1, 1, 1].    
    """
    return torch.sum(torch.abs(Z.view(Z.shape[0], -1)), 1)[:, None, None, None]
def l2_norm(x):
    return torch.norm(x, p=2)
def lp_norm(x, p = 2):
    return torch.norm(x, p=p)
def mean_perturbation(x, adv, dist):
    norms = []
    return torch.sum(dist(x, adv))/adv.shape[0]

def attack_success_rate_utgt(model, data_loader, device = None):
    """

    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total = 0
    success = 0
    
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Get model predictions for original and adversarial examples
        outputs = model(images)
        _, adv_predicted = torch.max(outputs.data, 1)

        # Count successful attacks
        total += labels.size(0)
        success += (adv_predicted != labels).sum().item()

    # Calculate success rate
    success_rate = 100 * success / total

    return success_rate