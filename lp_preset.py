def GeneralPattern(args):
    args.path = "~/Pictures/dataset/reid/"
    # this will create a folder named "_text_detection" under "~/Pictures/dataset/ocr"
    args.code_name = "_local_patch"
    # Set it to True to make experiment result reproducible
    args.deterministic_train = False
    # Random seed for everything
    # If deterministic_train is disabled, then it will have no meaning
    args.seed = 1
    # Training Hyperparameter
    args.learning_rate = 1e-4
    args.batch_size_per_gpu = 1
    args.loading_threads = 4
    args.img_channel = 3
    args.epoch_num = 200

    args.finetune = False

    # Image Normalization
    args.bg_color = 255

    args.augment_img = False
    args.img_mean = (0.5, 0.5, 0.5)
    args.img_std = (1.0, 1.0, 1.0)
    args.img_bias = (0.0, 0.0, 0.0)
    return args

def Unique_Patterns(args):
    args.train_sources = "OP_local_patch"
    # This determines the length of dataset
    args.batch_per_epoch = 100

    # Define composition of samples in a batch
    # 1 batch contains:
    # batch_common_person * n_cam_views * n_cam_views_same + batch_rare_person
    args.batch_common_person = 8
    args.n_cam_views = 3
    args.n_cam_views_same = 2
    args.batch_rare_person = 8

    args.ori_img_size = 40
    args.non_part_intensity = 128
    args.open_pose_joint_num = 17
    return args


def Runtime_Patterns(args):
    args.feature_comb = "add"
    args.model_prefix = "tmp"
    args.normalize_embedding = False
    args.global_embedding = False
    args.loss_weight = ["l2"]
    args.cfg_name = "default"
    return args


PRESET = {
    "general": GeneralPattern,
    "unique": Unique_Patterns,
    "runtime": Runtime_Patterns,
}