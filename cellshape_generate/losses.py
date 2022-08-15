import torch
try:
    from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
    chamfer_dist = chamfer_3DDist()
except:
    print("Chamfer3D not installed, using original chamfer distance")


def beta_loss(inputs, outputs, mu, log_var, kld_weight, criterion, beta):
    try:
        if isinstance(criterion, chamfer_3DDist):
            recon_loss = chamfer(inputs, outputs)
    except:
        recon_loss = criterion(inputs, outputs)

    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )

    loss = recon_loss + beta * kld_weight * kld_loss

    return loss, recon_loss, kld_loss


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)
