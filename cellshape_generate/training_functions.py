import torch
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

from cellshape_cloud.helpers.reports import print_log
from losses import beta_loss


def train(
    model, dataloader, num_epochs, criterion, optimizer, logging_info, kld_weight, beta
):

    name_logging, name_model, name_writer, name = logging_info

    writer = SummaryWriter(log_dir=name_writer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_loss = float("inf")
    niter = 1
    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.0
        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data[0]
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]

                # ===================forward=====================
                with torch.set_grad_enabled(True):
                    output, mu, log_var, z, feats = model(inputs)
                    optimizer.zero_grad()
                    loss, recon_loss, kld_loss = beta_loss(
                        inputs, output, mu, log_var, kld_weight, criterion, beta
                    )

                    # ===================backward====================
                    loss.backward()
                    optimizer.step()

                batch_loss = loss.detach().item() / batch_size
                batch_loss_recon = recon_loss.detach().item() / batch_size
                batch_loss_kld = kld_loss.detach().item() / batch_size
                running_loss += batch_loss
                batch_num += 1
                writer.add_scalar("/TotalLoss", batch_loss, niter)
                writer.add_scalar("/ReconLoss", batch_loss, niter)
                writer.add_scalar("/KLDLoss", batch_loss, niter)
                niter += 1
                tepoch.set_postfix(
                    loss=batch_loss,
                    recon_loss=batch_loss_recon,
                    kld_loss=batch_loss_kld,
                )

                if batch_num % 10 == 0:
                    logging.info(
                        f"[{epoch}/{num_epochs}]"
                        f"[{batch_num}/{len(dataloader)}]"
                        f"Total loss (recon + kld): {batch_loss} ({batch_loss_recon} + {batch_loss_kld})"
                    )

            total_loss = running_loss / len(dataloader)
            if total_loss < best_loss:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": total_loss,
                }
                best_loss = total_loss
                torch.save(checkpoint, name_model)
                logging.info(f"Saving model to {name_model} with loss = {best_loss}.")
                print(f"Saving model to {name_model} with loss = {best_loss}.")

        logging.info(f"Finished epoch {epoch} with loss={best_loss}.")
        print(f"Finished epoch {epoch} with loss={best_loss}.")
    print_log(f"Finished training {num_epochs} epochs.")
    return model, name_logging, name_model, name_writer, name
