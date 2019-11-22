import os, time, datetime, random, glob
import cv2, torch
import numpy as np
import lp_data as data
import lp_model as model
from lp_args import parse_arguments
import omni_torch.utils as util
import lp_preset as preset
from imgaug import augmenters
import matplotlib.pyplot as plt


args = parse_arguments()
opt = util.get_args(preset.PRESET)
args = util.cover_edict_with_argparse(opt, args)
torch.backends.cudnn.benchmark = True


def test(net, test_sample, gt_label):
    test_sample = test_sample.cuda()

    pred, attn = net(test_sample, test=True)
    if args.global_embedding:
        l_pred = pred.unsqueeze(0).repeat(pred.shape[0], 1, 1)  # * lr_attn
        r_pred = pred.unsqueeze(1).repeat(1, pred.shape[0], 1)  # * lr_attn
        dis = torch.sum((l_pred - r_pred), dim=-1)
    else:
        l_attn = attn.unsqueeze(0).repeat(attn.shape[0], 1, 1, 1)
        r_attn = attn.unsqueeze(1).repeat(1, attn.shape[0], 1, 1)
        lr_attn = l_attn * r_attn

        l_pred = pred.unsqueeze(0).repeat(pred.shape[0], 1, 1, 1)# * lr_attn
        r_pred = pred.unsqueeze(1).repeat(1, pred.shape[0], 1, 1)# * lr_attn
        dis = torch.mean(torch.sum((l_pred - r_pred), dim=-1), dim=-1)

    dis_img = (dis - dis.min()) / (dis.max() - dis.min()) * 255
    cv2.imwrite(os.path.expanduser("~/Pictures/dataset/reid/eval_lp/result.jpg"), dis_img.cpu().numpy())


def load_img(load_img_list, aug_seq):
    aug_seq = aug_seq.to_deterministic()
    batch = []
    bad_img_sum = 0
    for img_folder in load_img_list:
        imgs = []
        for i in range(args.open_pose_joint_num):
            path = os.path.join(img_folder, "%s.jpg" % str(i).zfill(2))
            if os.path.exists(path):
                imgs.append(cv2.imread(path))
            else:
                bad_img_sum += 1
                imgs.append(np.ones(
                    (args.ori_img_size, args.ori_img_size, args.img_channel)
                ).astype("uint8") * args.non_part_intensity)
        imgs = aug_seq.augment_images(imgs)
        imgs = np.concatenate(imgs, axis=2)
        batch.append(imgs)
    batch = torch.tensor(batch, dtype=torch.float)
    batch = batch.permute(0, 3, 1, 2)
    print("This batch has %d local patches which is empty" % (bad_img_sum))
    return batch


def augment():
    aug_list = []
    aug_list.append(augmenters.CropToFixedSize(width=32, height=32,
                                               position=(0.5, 0.5)))
    aug_list.append(augmenters.JpegCompression(compression=[0, 85]))
    aug_seq = augmenters.Sequential(aug_list, random_order=False)
    return aug_seq


def main():
    test_folder = os.path.expanduser("~/Pictures/dataset/reid/eval_lp/HHR_Body")
    diff_person_choose = 4
    diff_view = 8


    net = model.Encoder(args)
    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    net = util.load_latest_model(args, net, prefix=args.model_prefix, nth=1)


    #test_data, label_mapping = data.fetch_dataset(args, verbose=False, for_test=True)
    person_id = os.listdir(test_folder)
    gt_label, views = [], []
    for p_id in sorted(person_id):
        views += random.sample(glob.glob(os.path.join(test_folder, p_id, "*")), diff_view)

    test_imgs = load_img(views, augment())
    gt_label = torch.arange(len(person_id)).unsqueeze(-1).repeat(1, diff_view).view(-1).int()

    with torch.no_grad():
        test(net, test_imgs, gt_label)



if __name__ == "__main__":
    main()
