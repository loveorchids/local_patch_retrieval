import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Local Patch')
    ##############
    #        TRAINING        #
    ##############
    parser.add_argument(
        "-dt",
        "--deterministic_train",
        action="store_true",
        help="if this is turned on, then everything will be deterministic"
             "and the training process will be reproducible.",
    )
    parser.add_argument(
        "-en",
        "--epoch_num",
        type=int,
        help="Epoch number of the training",
        default=200
    )
    parser.add_argument(
        "-bpg",
        "--batch_size_per_gpu",
        type=int,
        help="batch size inside each GPU during training",
        default=4
    )
    parser.add_argument(
        "-lt",
        "--loading_threads",
        type=int,
        help="loading_threads correspond to each GPU during both training and validation, "
             "e.g. You have 4 GPU and set -lt 2, so 8 threads will be used to load data",
        default=3
    )
    parser.add_argument(
        "--feature_comb",
        type=str,
        default="add",
        help="method to combine local and global feature"
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="tmp",
        help="the prefix of model name"
    )
    parser.add_argument(
        "--normalize_embedding",
        action="store_true",
        help="Normalize the embedding of local patches or not",
    )
    parser.add_argument(
        "--global_embedding",
        action="store_true",
        help="Generate global embedding of local patches or not",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
    )
    parser.add_argument(
        "--loss_weight",
        nargs='+',
        help="the loss weight for 3 set and 3 loss, length must be either 9 or 1, "
             "when length is 1, it should be a string",
        default=["l2"]
    )
    parser.add_argument(
        "--batch_common_person",
        type=int,
        default=6
    )
    parser.add_argument(
        "--n_cam_views",
        type=int,
        default=4
    )
    parser.add_argument(
        "--n_cam_views_same",
        type=int,
        default=2
    )
    parser.add_argument(
        "--batch_rare_person",
        type=int,
        default=4
    )

    args = parser.parse_args()
    return args