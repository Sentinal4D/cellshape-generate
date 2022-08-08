import torch


def beta_loss(inputs, outputs, mu, log_var, kld_weight, criterion, beta):
    recon_loss = criterion(inputs, outputs)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    loss = recon_loss + beta * kld_weight * kld_loss

    return loss
