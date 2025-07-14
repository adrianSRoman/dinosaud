import torch
from torch import nn
import torch.nn.functional as F

def mse_loss():
    return torch.nn.MSELoss()

def l1_loss():
    return torch.nn.L1Loss()

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def gradient_penalty(wave, output, weight = 10, center = 0.):
    batch_size, device = wave.shape[0], wave.device

    gradients = torch_grad(
        outputs = output,
        inputs = wave,
        grad_outputs = torch.ones_like(output),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((vector_norm(gradients, dim = 1) - center) ** 2).mean()
