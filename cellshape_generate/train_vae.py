import torch
from torch.utils.data import DataLoader
from datetime import datetime
import logging


from cellshape_cloud.vendor.chamfer_distance import ChamferLoss
from cellshape_cloud.pointcloud_dataset import PointCloudDataset, SingleCellDataset
from cloud_vae import CloudVAE
from cellshape_cloud.helpers.reports import get_experiment_name
from training_functions import train


def train_vae(args):
    autoencoder = CloudVAE(
        num_features=args.num_features,
        k=args.k,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
    )
    everything_working = True
    file_not_found = False
    wrong_architecture = False
    try:
        checkpoint = torch.load(args.pretrained_path)
    except FileNotFoundError:
        print(
            "This model doesn't exist."
            " Please check the provided path and try again. "
            "Ignore this message if you do not have a pretrained model."
        )
        checkpoint = {"model_state_dict": None}
        file_not_found = True
        everything_working = False
    except AttributeError:
        print("No pretrained model given.")
        checkpoint = {"model_state_dict": None}
        everything_working = False
    except:
        print("No pretrained model given.")
        checkpoint = {"model_state_dict": None}
        everything_working = False
    try:
        autoencoder.load_state_dict(checkpoint["model_state_dict"])
        print(f"The loss of the loaded model is {checkpoint['loss']}")
    except RuntimeError:
        print(
            "The model architecture given doesn't " "match the one provided."
        )
        print("Training from scratch")
        wrong_architecture = True
        everything_working = False
    except AttributeError or TypeError:
        print("Training from scratch")
    except:
        print("Training from scratch")

    try:
        model_dict = autoencoder.state_dict()  # load parameters from pre-trained FoldingNet
        for k in checkpoint['model_state_dict']:
            if k in model_dict:
                model_dict[k] = checkpoint['model_state_dict'][k]
                print("Found weight: " + k)

        autoencoder.load_state_dict(model_dict)

    except Exception as e:
        print(e)
        print("Tried loading some weights from a pre-trained autoencoder. Did not work")
    if args.dataset_type == "SingleCell":
        dataset = SingleCellDataset(
            args.dataframe_path, args.cloud_dataset_path
        )
    else:
        dataset = PointCloudDataset(args.cloud_dataset_path)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    reconstruction_criterion = ChamferLoss()

    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=args.learning_rate_autoencoder * 16 / args.batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    )
    logging_info = get_experiment_name(
        model=autoencoder, output_dir=args.output_dir
    )
    name_logging, name_model, name_writer, name = logging_info
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.basicConfig(filename=name_logging, level=logging.INFO)

    if everything_working:
        logging.info(
            f"Started training cluster model {name} at {now} "
            f"using autoencoder which is "
            f"saved at {args.pretrained_path}."
        )
        print(
            f"Started training model {name} at {now}."
            f"using autoencoder which is s"
            f"aved at {args.pretrained_path}."
        )
    if file_not_found:
        logging.info(
            f"The autoencoder model at {args.pretrained_path}"
            f" doesn't exist."
            f"if you knew this already, then don't worry. "
            f"If not, then check the path and try again"
        )
        logging.info("Training from scratch")
        print(
            f"The autoencoder model at "
            f"{args.pretrained_path} doesn't exist. "
            f"If you knew this already, then don't worry. "
            f"If not, then check the path and try again"
        )
        print("Training from scratch")

    if wrong_architecture:
        logging.info(
            f"The autoencoder model at {args.pretrained_path} has "
            f"a different architecture to the one provided "
            f"If not, then check the path and try again"
        )
        logging.info("Training from scratch")
        print(
            f"The autoencoder model at {args.pretrained_path} "
            f"has a different architecture to the one provided "
            f"If not, then check the path and try again."
        )
        print("Training from scratch")

    for arg, value in sorted(vars(args).items()):
        logging.info(f"Argument {arg}: {value}")
        print(f"Argument {arg}: {value}")

    autoencoder, name_logging, name_model, name_writer, name = train(
        model=autoencoder,
        dataloader=dataloader,
        num_epochs=args.num_epochs_autoencoder,
        criterion=reconstruction_criterion,
        optimizer=optimizer,
        logging_info=logging_info,
        kld_weight=args.kld_weight,
        beta=args.beta
    )
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"Finished training at {now}.")
    print(f"Finished training at {now}.")

    return autoencoder, name_logging, name_model, name_writer, name
