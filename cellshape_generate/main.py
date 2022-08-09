import argparse

from train_vae import train_vae


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cellshape-cloud")
    parser.add_argument(
        "--cloud_convert",
        default=False,
        type=str2bool,
        help="Do you need to convert 3D images to point clouds?",
    )

    parser.add_argument(
        "--tif_dataset_path",
        default="/home/mvries/Documents/CellShape/"
        "UploadData/Dataset/TestConvert/TestTiff/",
        type=str,
        help="Please provide the path to the " "dataset of 3D tif images",
    )
    parser.add_argument(
        "--mesh_dataset_path",
        default="/home/mvries/Documents/CellShape/"
        "UploadData/Dataset/TestConvert/TestMesh/",
        type=str,
        help="Please provide the path to the " "dataset of 3D meshes.",
    )
    parser.add_argument(
        "--cloud_dataset_path",
        default="/home/mvries/Documents/Datasets/"
                "OPM/SingleCellFromNathan_17122021/",
        type=str,
        help="Please provide the path to the " "dataset of the point clouds.",
    )
    parser.add_argument(
        "--dataset_type",
        default="SingleCell",
        type=str,
        choices=["SingleCell", "Other"],
        help="Please provide the type of dataset. "
        "If using the one from our paper, then choose 'SingleCell', "
        "otherwise, choose 'Other'.",
    )
    parser.add_argument(
        "--dataframe_path",
        default="/home/mvries/Documents/Datasets/"
                "OPM/SingleCellFromNathan_17122021"
                "/all_data_removedwrong_ori_removedTwo.csv",
        type=str,
        help="Please provide the path to the dataframe "
        "containing information on the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/mvries/Documents/Testing_output_vae/",
        type=str,
        help="Please provide the path for where to save output.",
    )
    parser.add_argument(
        "--num_epochs_autoencoder",
        default=250,
        type=int,
        help="Provide the number of epochs for the autoencoder training.",
    )
    parser.add_argument(
        "--num_features",
        default=128,
        type=int,
        help="Please provide the number of " "features to extract.",
    )
    parser.add_argument(
        "--k", default=20, type=int, help="Please provide the value for k."
    )
    parser.add_argument(
        "--encoder_type",
        default="dgcnn",
        type=str,
        help="Please provide the type of encoder.",
    )
    parser.add_argument(
        "--decoder_type",
        default="foldingnetbasic",
        type=str,
        help="Please provide the type of decoder.",
    )
    parser.add_argument(
        "--learning_rate_autoencoder",
        default=0.0001,
        type=float,
        help="Please provide the learning rate "
        "for the autoencoder training.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Please provide the batch size.",
    )
    parser.add_argument(
        "--proximal",
        default=0,
        type=int,
        help="Please provide the value of proximality "
        "[0 = distal, 1 = proximal, 2 = both].",
    )
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="Please provide the path to a pretrained autoencoder.",
    )

    parser.add_argument(
        "--beta",
        default=4,
        type=int,
        help="Please provide a value for beta for the beta-vae"
             ". See https://openreview.net/forum?id=Sy2fzU9gl.",
    )
    parser.add_argument(
        "--kld_weight",
        default=1,
        type=int,
        help="Please provide a value for Kullback_liebler convergence weight"
             ". See https://openreview.net/forum?id=Sy2fzU9gl.",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Random seed.",
    )
    args = parser.parse_args()
    output = train_vae(args)
